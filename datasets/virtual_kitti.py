from pathlib import Path

import torch
from torch.utils.data import random_split

from .coco import CocoDetection
import datasets.transforms as T

import ipdb


def make_vkitti_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)

    assert root.exists(), f'provided VirtualKitti path {root} does not exist'
    dataset = CocoDetection(root / "data", root / "labels.json", transforms=make_vkitti_transforms(image_set), return_masks=False)
    train_ratio = 0.7

    n_train = int(len(dataset)*train_ratio)
    lengths = [n_train, len(dataset) - n_train]
    train_dataset, val_dataset = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    return dataset
