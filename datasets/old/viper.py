from pathlib import Path

import torch
from torch.utils.data import random_split

from .coco import CocoDetection
import datasets.transforms as T

import ipdb


def make_viper_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    scales = [(800, 250), (900, 300), (1000, 300), (1100, 350)]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                #T.RandomResize(scales, max_size=1333),
                T.RandomSizeCropWH(min_size=scales[0], max_size=scales[-1]), 
                T.Compose([
                    #T.RandomResize([400, 500, 600]),
                    T.RandomResize([(1600, 750), (1500, 700), (1200, 550)]), 
                    #T.RandomSizeCrop(384, 600),
                    T.RandomSizeCropWH(min_size=(1000, 300), max_size=(1000, 300)), 
                    #T.RandomResize(scales, max_size=1333),
                    T.RandomResize(scales),
                ])
            ),
            normalize,
        ])


    if image_set == 'val':
        return T.Compose([
            #T.RandomResize([800], max_size=1333),
            T.RandomSizeCropWH(min_size=(1000, 300), max_size=(1000, 300)), 
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)

    assert root.exists(), f'provided VIPER path {root} does not exist'

    PATH = {
        "train": (root / "train" / "data", root / "train" / "labels.json"), 
        "val": (root / "val" / "data", root / "val" / "labels.json"), 
    }
    dataset = CocoDetection(PATH[image_set][0], PATH[image_set][1], transforms=make_viper_transforms(image_set), return_masks=False)
    
    return dataset
