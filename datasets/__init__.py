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



def build_evaluator(args, postprocessors, dataset_val):
    
    if args.dataset_file == 'coco':
        from .coco_eval import CocoEvaluator
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        coco_evaluator = CocoEvaluator(base_ds, iou_types)
        # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    elif args.dataset_file == 'kitti':
        from .kitti_eval import KittiEvaluator
        iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
        return KittiEvaluator(dataset_val, iou_types)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')
