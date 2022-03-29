"""
Refer here kitti labels: https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt
"""
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
#from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

import torch
from torch.utils.data import random_split
import torchvision

from .kitti import KITTI_CLASSES
from pycocotools.coco import COCO
import datasets.transforms as T
import ipdb



VIPER_TO_KITTI = {
    "unlabeled": "Misc", 
    "ambiguous": "Misc", 
    "sky": "Misc", 
    "road": "Misc", 
    "sidewalk": "Misc", 
    "railtrack": "Misc", 
    "terrain": "Misc", 
    "tree": "Misc", 
    "vegetation": "Misc", 
    "building": "Misc", 
    "infrastructure": "Misc", 
    "fence": "Misc", 
    "billboard": "Misc", 
    "trafficlight": "Misc", 
    "trafficsign": "Misc", 
    "mobilebarrier": "Misc", 
    "firehydrant": "Misc", 
    "chair": "Misc", 
    "trash": "Misc", 
    "trashcan": "Misc", 
    "person": "Pedestrian",     # KITTI has Pedestrian and Person_sitting
    "animal": "Misc", 
    "bicycle": "Cyclist", 
    "motorcycle": "Cyclist", 
    "car": "Car", 
    "van": "Van", 
    "bus": "Truck",             # KITTI does not have bus
    "truck": "Truck", 
    "trailer": "Misc",
    "train": "Tram", 
    "plane": "Misc", 
    "boat": "Misc"
}

VIPER_CLASSES = pd.read_csv(Path(__file__).parent / 'viper_classes.csv')["classname"]
VIPER_CLASSES = np.array(VIPER_CLASSES).tolist()


class PrepareVIPER(object):

    def __call__(self, img, target):

        img_w, img_h = img.size
        img_id = target["image_id"]
        img_id = torch.tensor(img_id)

        anno = target["annotations"]
        anno = decode_viper_anno(anno)

        boxes = []
        for box in anno["two_d_bbox"]:
            left, top, right, bottom = box
            x_top_left, y_top_left, width, height = left, top, right - left,  bottom - top
            boxes.append([x_top_left, y_top_left, width, height])

            
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=img_w)
        boxes[:, 1::2].clamp_(min=0, max=img_h)


        classes = [KITTI_CLASSES.index(VIPER_TO_KITTI[obj]) for obj in anno['class_name']]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = img_id
        
        target["orig_size"] = torch.as_tensor([int(img_h), int(img_w)])
        target["size"] = torch.as_tensor([int(img_h), int(img_w)])
        
        return img, target


def _type_to_category_id(name):
    return VIPER_CLASSES.index(name)


def decode_viper_anno(anno):
    if len(anno.shape) == 1:
        anno = anno.rehsape(1, -1)
        
    #instance_id = np.array(anno[:, 0]).astype(int)
    
    class_id = np.array(anno[:, 1]).astype(int)
    two_d_bbox = np.array(anno[:, 2:6]).astype(int)
    #three_d_bbox_coordinate = np.array(anno[:, 6:12]).astype(float)
    #three_d_bbox_matrix = np.array(anno[:, 12:28]).astype(float)

    return {
        "class_id": class_id, 
        "class_name": np.array([VIPER_CLASSES[i] for i in class_id]), 
        "two_d_bbox": two_d_bbox
    }
    

class VIPERDetection(torchvision.datasets.VisionDataset):
    
    #_mean = [0.3482, 0.3424, 0.3392]
    #_std = [0.2448, 0.2404, 0.2379]
    
    def __init__(self, root, seed, transforms, **kwargs):
        
        super(VIPERDetection, self).__init__(root, **kwargs)


        self.coco_labels = COCO(osp.join(root, "bb/labels_iscrowd.json"))
        
        labels = json.load(open(osp.join(root, "bb/labels_iscrowd.json")))
        self.filename_to_id = {}
        for i in labels["images"]:
            self.filename_to_id[i["file_name"]] = i["id"]
        
        self.seed = seed
        self.prepare = PrepareVIPER()
        self.img_paths = self.load_img_paths()
        self._transforms = transforms
        
    def load_img_paths(self):
                 
        img_paths = []
        for img_path in tqdm(self.root.glob("img/*/*.png"), desc=f"Checking for valid data"):
            bbox_path = img_path.parent.parent.parent / "bb" / img_path.parent.name / img_path.name.replace("png", "csv")
            if bbox_path.exists():
                img_paths.append(img_path)

        return img_paths
    
    def __getitem__(self, index):
        
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        bbox_path = img_path.parent.parent.parent / "bb" / img_path.parent.name / img_path.name.replace("png", "csv")
        bboxes = np.array(pd.read_csv(bbox_path, header=None))
        
        img_id = self.filename_to_id[img_path.name]
        img, target = self.prepare(img, {"image_id": img_id, "annotations": bboxes})
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target
    
    def __len__(self):
        return len(self.img_paths)
        
        
def make_viper_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    elif image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.root_path)
    assert root.exists(), f'provided VIPER path {root} does not exist'
    
    dataset = VIPERDetection(root / image_set, args.seed, transforms=make_viper_transforms(image_set))
    return dataset
