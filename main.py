# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import hydra
import datetime
import json
import random
import time
import datetime
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset, get_class_mapping
from engine import evaluate, train_one_epoch
from models import build_model


import ipdb
import logging
logger = logging.getLogger(__name__)



@hydra.main(config_path="./config", config_name="base")
def main(args):
    
    logger.info(OmegaConf.to_yaml(args))
    logger.info(f'Current working directory: {os.getcwd()}')
    OmegaConf.save(config=args, f='config.yaml')
    

    utils.init_distributed_mode(args)
    logger.info("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights:
        assert args.dataset.masks, "Frozen training is meant for segmentation only"
    

    if args.use_wandb and utils.is_main_process():
        import wandb
        ct = datetime.datetime.now()
        wandb.init(
            project=f"label-translation-detr-{args.dataset_file}", 
            entity="andrew-liao", 
            name=f"{ct.year}.{ct.month}.{ct.day}.{ct.hour}.{ct.minute}.{ct.second}", 
            config=args
        )

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    

    dataset_train = build_dataset(image_set='train', dataset_args=args.dataset)
    dataset_val = build_dataset(image_set='val', dataset_args=args.dataset)
    

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)


    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, 
                                   num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers)



    base_ds = get_coco_api_from_dataset(dataset_val)

    
    
    
    if args.dataset.name != args.add_dataset.name and args.add_dataset.name != "dummy":
        class_mapping = get_class_mapping(dataset_train)
        add_dataset_val = build_dataset(image_set='val', dataset_args=args.add_dataset, given_class_mapping=class_mapping)
        if args.distributed:
            add_sampler_val = DistributedSampler(add_dataset_val, shuffle=False)
        else:
            add_sampler_val = torch.utils.data.SequentialSampler(add_dataset_val)

        add_data_loader_val = DataLoader(add_dataset_val, args.batch_size, sampler=add_sampler_val,
                                 drop_last=False, 
                                 collate_fn=utils.collate_fn, 
                                 num_workers=args.num_workers)
        
        add_base_ds = get_coco_api_from_dataset(add_dataset_val)
    

    if args.frozen_weights:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])



    output_dir = Path(".")
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval_only and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    
    if args.eval_only:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device)
        
        utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    
    
    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        
        

    
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        # extra checkpoint before LR drop and every 100 epochs
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
            checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)
        
        
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device
        )

        log_stats = {**{f'train/{k}': v for k, v in train_stats.items()},
                     **{f'test/{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        
        if args.dataset.name != args.add_dataset.name and args.add_dataset.name != "dummy":
            add_test_stats, add_coco_evaluator = evaluate(
                model, criterion, postprocessors, add_data_loader_val, add_base_ds, device
            )
            log_stats.update({f'add_test/{k}': v for k, v in add_test_stats.items()})


        if args.use_wandb and utils.is_main_process():
            wandb.log(log_stats)
            wandb.watch(model)
            
        
        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    main()
