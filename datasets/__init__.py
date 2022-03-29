# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .viper import VIPERDetection


def get_coco_api_from_dataset(dataset_val):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset_val, torch.utils.data.Subset):
            dataset_val = dataset_val.dataset
        
    if isinstance(dataset_val, torchvision.datasets.CocoDetection):
        return dataset_val.coco
    elif isinstance(dataset_val, torchvision.datasets.Kitti):
        return dataset_val.coco_labels
    
    elif isinstance(dataset_val, VIPERDetection):
        return dataset_val.coco_labels


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        from .coco import build as build_coco
        return build_coco(image_set, args)
    elif args.dataset_file == 'viper':
        from .viper import build as build_viper
        return build_viper(image_set, args)
    elif args.dataset_file == 'kitti':
        from .kitti import build as build_kitti
        return build_kitti(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')
