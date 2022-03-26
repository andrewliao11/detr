# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision


def get_dataset_class(args):
    if args.dataset_file == 'coco':
        from .coco import CocoDetection
        return CocoDetection
    elif args.dataset_file == 'coco_panoptic':
        from .coco_panoptic import CocoPanoptic
        # to avoid making panopticapi required for coco
        return CocoPanoptic
    elif args.dataset_file == 'viper':
        from .viper import ViperDetection
        return ViperDetection
    elif args.dataset_file == 'kitti':
        from .kitti import KittiDetection
        return KittiDetection
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        from .coco import build as build_coco
        return build_coco(image_set, args)
    elif args.dataset_file == 'coco_panoptic':
        from .coco_panoptic import build as build_coco_panoptic
        # to avoid making panopticapi required for coco
        return build_coco_panoptic(image_set, args)
    elif args.dataset_file == 'viper':
        from .viper import build as build_viper
        return build_viper(image_set, args)
    elif args.dataset_file == 'kitti':
        from .kitti import build as build_kitti
        return build_kitti(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')
