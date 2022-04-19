"""
potential reference for augmentation: https://github.com/QiuJueqin/SqueezeDet-PyTorch/blob/master/src/datasets/base.py
"""
from pathlib import Path

import os
import torch
from torch.utils.data import random_split

from .coco import CocoDetection
import datasets.transforms as T

import ipdb


def make_cityscapes_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSizeCropWH(min_size=(1280, 640), max_size=(1440, 720)),
            normalize,
        ])


    if image_set == 'val':
        return T.Compose([
            T.RandomSizeCropWH(min_size=(1280, 640), max_size=(1440, 720)),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, dataset_args, given_class_mapping=None):
    root = os.environ['HOME'] / Path(dataset_args.path)

    assert root.exists(), f'provided cityscapes path {root} does not exist'
    dataset = CocoDetection(root / image_set / "data", root / image_set / "labels.json", 
                            transforms=make_cityscapes_transforms(image_set), return_masks=False, given_class_mapping=given_class_mapping)
    
    return dataset
