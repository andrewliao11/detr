# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision


def get_coco_api_from_dataset(dataset_val):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset_val, torch.utils.data.Subset):
            dataset_val = dataset_val.dataset
        
    if isinstance(dataset_val, torchvision.datasets.CocoDetection):
        return dataset_val.coco


def build_dataset(image_set, args):
    if args.dataset.name in ['mscoco14', 'mscoco17']:
        from .coco import build as build_coco
        return build_coco(image_set, args)
    elif args.dataset.name == 'virtual_kitti':
        from .virtual_kitti import build as build_vkitti
        return build_vkitti(image_set, args)
    elif args.dataset.name == 'viper':
        from .viper import build as build_viper
        return build_viper(image_set, args)
    elif args.dataset.name == 'kitti':
        from .kitti import build as build_kitti
        return build_kitti(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset.name} not supported')
