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


def get_class_mapping(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
        
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return {d["id"]: d["name"].lower() for d in dataset.coco.dataset["categories"]}


def build_dataset(image_set, dataset_args, given_class_mapping=None):
    if dataset_args.name in ['mscoco14', 'mscoco17']:
        from .coco import build as build_coco
        return build_coco(image_set, dataset_args, given_class_mapping=given_class_mapping)
    elif dataset_args.name == 'virtual_kitti':
        from .virtual_kitti import build as build_vkitti
        return build_vkitti(image_set, dataset_args, given_class_mapping=given_class_mapping)
    #elif dataset_args.name == 'viper':
    #    from .viper import build as build_viper
    #    return build_viper(image_set, dataset_args, given_class_mapping=given_class_mapping)
    elif dataset_args.name == 'kitti':
        from .kitti import build as build_kitti
        return build_kitti(image_set, dataset_args, given_class_mapping=given_class_mapping)
    elif dataset_args.name == 'mixed_kitti_virtual_kitti':
        from .mixed_kitti_virtual_kitti import build as build_mixed_kitti_virtual_kitti
        return build_mixed_kitti_virtual_kitti(image_set, dataset_args, given_class_mapping=given_class_mapping)
    elif dataset_args.name == 'synscapes':
        from .synscapes import build as build_synscapes
        return build_synscapes(image_set, dataset_args, given_class_mapping=given_class_mapping)
    elif dataset_args.name == 'cityscapes':
        from .cityscapes import build as build_cityscapes
        return build_cityscapes(image_set, dataset_args, given_class_mapping=given_class_mapping)
    else:
        raise ValueError(f'dataset {dataset_args.name} not supported')
