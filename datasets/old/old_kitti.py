"""
Refer here kitti labels: https://github.com/bostondiditeam/kitti/blob/master/resources/devkit_object/readme.txt
"""
import os
import json
import random
import os.path as osp
from pathlib import Path

import torch
from torch.utils.data import random_split
import torchvision

from pycocotools.coco import COCO
import datasets.transforms as T
import ipdb


KITTI_CLASSES = set(["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"])

class PrepareKitti(object):
    def __call__(self, img, target):

        img_w, img_h = img.size
        img_id = target["image_id"]
        img_id = torch.tensor([img_id])

        anno = target["annotations"]

        boxes = []
        for obj in anno:
            # KITTI bounding box: left, top, right, bottom 
            left, top, right, bottom = obj["bbox"]
            x_top_left, y_top_left, width, height = left, top, right - left,  bottom - top
            boxes.append([x_top_left, y_top_left, width, height])

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=img_w)
        boxes[:, 1::2].clamp_(min=0, max=img_h)


        # warning: doing this is incorrect
        classes = [_type_to_category_id(obj["type"]) for obj in anno]
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


# Things didn't check
# 2. ground truth representations
# 3. If the transformed aspect ratio makes sense -> do not force it to be square
class KittiDetection(torchvision.datasets.Kitti):
    
    def __init__(self, root, image_set, transforms):
        super(KittiDetection, self).__init__(root=root, train=image_set == "training")

        self.coco_labels = COCO(osp.join(root, f"Kitti/raw/{image_set}/labels_iscrowd.json"))
        
        # warning: does does not match
        labels = json.load(open(osp.join(root, f"Kitti/raw/{image_set}/labels_iscrowd.json")))
        
        self.filename_to_id = {}
        for i in labels["images"]:
            self.filename_to_id[i["file_name"]] = i["id"]
            
        self._transforms = transforms
        self.prepare = PrepareKitti()
    
    @property
    def _raw_folder(self):
        return os.path.join(self.root, "Kitti", "raw")
    
    def __getitem__(self, idx):
        while True:
            try:
                img, target = super(KittiDetection, self).__getitem__(idx)
                img = img.convert("RGB")
                break
            except OSError:
                print(f"Cannot load {self.images[idx]}")
                idx = random.choices(range(len(self)))[0]

        img_path = self.images[idx]
        filename = Path(img_path).name
        img_id = self.filename_to_id[filename]
        
        img, target = self.prepare(img, {"image_id": img_id, "annotations": target})
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
    

def make_kitti_transforms(image_set):

    # imagenet mean and stddev
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    #Resize to square does not make sense
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
    assert root.exists(), f'provided KITTI path {root} does not exist'
    
    dataset = KittiDetection(root, "training", transforms=make_kitti_transforms(image_set))
    
    train_ratio = 0.7
    
    n_train = int(len(dataset)*train_ratio)
    lengths = [n_train, len(dataset) - n_train]
    train_dataset, val_dataset = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))
    if image_set == "train":
        return train_dataset
    if image_set == "val":
        return val_dataset
    
