"""
potential reference for augmentation: https://github.com/QiuJueqin/SqueezeDet-PyTorch/blob/master/src/datasets/base.py
"""
import os
from pathlib import Path

import torch
from torch.utils.data import random_split

from .coco import CocoDetection
import datasets.transforms as T

import ipdb


def make_mixed_kitti_virtual_kitti_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
#            T.RandomResize((1240, 375)),
            normalize,
        ])


    if image_set == "val":
        return T.Compose([
#            T.RandomResize((1240, 375)),
            normalize,
        ])

    raise ValueError(f"unknown {image_set}")


def build(image_set, dataset_args, given_class_mapping=None):
    root = os.environ["HOME"] / Path(dataset_args.path)

    assert root.exists(), f"provided Mixed Kitti-Virtual Kitti path {root} does not exist"
    dataset = CocoDetection(root / image_set / "data", root / image_set / "labels.json", transforms=make_mixed_kitti_virtual_kitti_transforms(image_set), return_masks=False, given_class_mapping=given_class_mapping)
    return dataset
