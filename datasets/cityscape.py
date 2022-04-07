"""
potential reference for augmentation: https://github.com/QiuJueqin/SqueezeDet-PyTorch/blob/master/src/datasets/base.py
"""
from pathlib import Path

import torch
from torch.utils.data import random_split

from .coco import CocoDetection
import datasets.transforms as T

import ipdb


def make_kitti_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            normalize,
        ])


    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.dataset.path)

    assert root.exists(), f'provided Kitti path {root} does not exist'
    dataset = CocoDetection(root / "data", root / "labels.json", transforms=make_kitti_transforms(image_set), return_masks=False)
    train_ratio = 0.7

    n_train = int(len(dataset)*train_ratio)
    lengths = [n_train, len(dataset) - n_train]
    train_dataset, val_dataset = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    return dataset
