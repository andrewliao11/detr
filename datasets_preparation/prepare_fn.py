import os
import json
import random
import imagesize
import copy
import pandas as pd

from tqdm import tqdm
from PIL import Image
from pathlib import Path

# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
from cityscape_labels import labels as CITYSCAPES_LABELS
        
import ipdb

SEED = 42


class BaseCOCOPrepare():
    def __init__(self, dataset_name, dataset_root):
        self.dataset_name = dataset_name
        self.dataset_root = Path(dataset_root) if isinstance(dataset_root, str) else dataset_root
        self.target_dataset_root = os.environ["HOME"] / Path(f"datasets/{self.dataset_name}/coco_format")
        os.makedirs(self.target_dataset_root, exist_ok=True)

    def prepare_images(self):
        raise NotImplementedError

    def prepare(self):
        
        self.prepare_splits_if_needed()
        self.prepare_images()
        self.prepare_labels()

    def prepare_splits_if_needed(self):
        pass

    def _scale_bbox(self, bbox, image_w, image_h):
        ratio, direction = self.scale
        assert ratio >= 0., "ratio need to be positive"
        assert direction in ["up", "down"], "direction need to be either up or down"
        
        x_top_left, y_top_left, bbox_width, bbox_height = bbox
        
        x_bottom_right = x_top_left + bbox_width
        y_bottom_right = y_top_left + bbox_height
        
        x_center = (x_top_left + x_bottom_right) / 2.
        y_center = (y_top_left + y_bottom_right) / 2.
        
        if direction == "up":
            y_top_left = y_center - bbox_height * (1+ratio) / 2.
            y_bottom_right = y_center + bbox_height * (1+ratio) / 2.
            
            x_top_left = x_center - bbox_width * (1+ratio) / 2.
            x_bottom_right = x_center + bbox_width * (1+ratio) / 2.
            
            y_top_left = max(0, y_top_left)
            y_bottom_right = min(image_h, y_bottom_right)
            
            x_top_left = max(0, x_top_left)
            x_bottom_right = min(image_w, x_bottom_right)
            
        if direction == "down":
            y_top_left = y_center - bbox_height * (1-ratio) / 2.
            y_bottom_right = y_center + bbox_height * (1-ratio) / 2.
            
            x_top_left = x_center - bbox_width * (1-ratio) / 2.
            x_bottom_right = x_center + bbox_width * (1-ratio) / 2.
            
            y_top_left = max(0, y_top_left)
            y_bottom_right = min(image_h, y_bottom_right)
            
            x_top_left = max(0, x_top_left)
            x_bottom_right = min(image_w, x_bottom_right)
            
        new_bbox_width = x_bottom_right - x_top_left
        new_bbox_height = y_bottom_right - y_top_left
        return [x_top_left, y_top_left, new_bbox_width, new_bbox_height]

    def _shift_bbox(self, bbox, image_w, image_h):
        ratio, direction = self.shift
        assert ratio >= 0., "ratio need to be positive"
        assert direction in ["left", "right", "up", "down"], "direction need to be either left, right, top, bottom"
        x_top_left, y_top_left, bbox_width, bbox_height = bbox
        
        x_bottom_right = x_top_left + bbox_width
        y_bottom_right = y_top_left + bbox_height
        
        if direction == "up":
            offset = ratio * bbox_height
            y_top_left = max(0, y_top_left - offset)
            y_bottom_right -= offset
            
        if direction == "down":
            offset = ratio * bbox_height
            y_top_left += offset
            y_bottom_right = min(image_h, y_bottom_right + offset)
            
        if direction == "left":
            offset = ratio * bbox_width
            
            x_top_left = max(0, x_top_left - offset)
            x_bottom_right -= offset
            
        if direction == "right":
            offset = ratio * bbox_width
            x_top_left += offset
            x_bottom_right = min(image_w, x_bottom_right + offset)
            
            
        new_bbox_height = y_bottom_right - y_top_left
        new_bbox_width = x_bottom_right - x_top_left
            
        return [x_top_left, y_top_left, new_bbox_width, new_bbox_height]

    def shift_or_scale_bbox(self, bbox, image_w, image_h):
        if self.shift != "no":
            bbox = self._shift_bbox(bbox, image_w, image_h)

        if self.scale != "no":
            bbox = self._scale_bbox(bbox, image_w, image_h)

        return bbox

    def decode_shift_and_scale_str(self, shift, scale):

        if shift != "no":
            ratio, direction = shift.split("-")
            ratio = float(ratio)
            shift = (ratio, direction)

        if scale != "no":
            ratio, direction = scale.split("-")
            ratio = float(ratio)
            scale = (ratio, direction)

        self.shift = shift
        self.scale = scale

    def change_dataset_name(self, dataset_name):
        if self.shift != "no":
            ratio, direction = self.shift
            dataset_name += f"_shift-{ratio}-{direction}"

        if self.scale != "no":
            
            ratio, direction = self.scale
            dataset_name += f"_scale-{ratio}-{direction}"

        return dataset_name


class Mscoco14Prepare(BaseCOCOPrepare):
    def __init__(self):
        dataset_name = "mscoco14"
        dataset_root = "/datasets/mscoco14"
        super(Mscoco14Prepare, self).__init__(dataset_name, dataset_root)
    
    def prepare_images(self):

        os.makedirs(self.target_dataset_root / "train", exist_ok=True)
        
        os.chdir(self.target_dataset_root / "train")
        cmd = f"ln -s {self.dataset_root / 'train2014/train2014'} data"
        
        if not Path("data").exists():
            os.system(cmd)

        os.makedirs(self.target_dataset_root / "val", exist_ok=True)
        os.chdir(self.target_dataset_root / "val")
        cmd = f"ln -s {self.dataset_root / 'val2014/val2014'} data"
        
        if not Path("data").exists():
            os.system(cmd)

    def prepare_labels(self):
        os.chdir(self.target_dataset_root / "train")
        cmd = f"ln -s {self.dataset_root / 'instances_train-val2014/annotations/instances_train2014.json'} labels.json"
        
        if not Path("labels.json").exists():
            os.system(cmd)

        os.chdir(self.target_dataset_root / "val")
        cmd = f"ln -s {self.dataset_root / 'instances_train-val2014/annotations/instances_val2014.json'} labels.json"
        if not Path("labels.json").exists():
            os.system(cmd)


class Mscoco17Prepare(BaseCOCOPrepare):
    def __init__(self):
        dataset_name = "mscoco17"
        dataset_root = "/datasets/mscoco17"
        super(Mscoco17Prepare, self).__init__(dataset_name, dataset_root)

    def prepare_images(self):

        os.makedirs(self.target_dataset_root / "train", exist_ok=True)
        os.chdir(self.target_dataset_root / "train")
        cmd = f"ln -s {self.dataset_root / 'train2017'} data"
        
        if not Path("data").exists():
            os.system(cmd)
            

        os.makedirs(self.target_dataset_root / "val", exist_ok=True)
        os.chdir(self.target_dataset_root / "val")
        cmd = f"ln -s {self.dataset_root / 'val2017'} data"
        if not Path("data").exists():
            os.system(cmd)
            

    def prepare_labels(self):

        os.chdir(self.target_dataset_root / "train")
        cmd = f"ln -s {self.dataset_root / 'annotations/instances_train2017.json'} labels.json"
        if not Path("labels.json").exists():
            os.system(cmd)


        os.chdir(self.target_dataset_root / "val")

        cmd = f"ln -s {self.dataset_root / 'annotations/instances_val2017.json'} labels.json"
        if not Path("labels.json").exists():
            os.system(cmd)


class KittiPrepare(BaseCOCOPrepare):
    KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]

    def __init__(self, shift, scale, train_ratio=0.7, image_prefix=""):

        dataset_name = "kitti"
        dataset_root = "/datasets/kitti"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)
        
        super(KittiPrepare, self).__init__(dataset_name, dataset_root)

        self.train_ratio = train_ratio
        self.image_prefix = image_prefix

    def _convert_source_path_to_target_name(self, src_p):
        return f"{self.image_prefix}{src_p.name}"

    def prepare_splits_if_needed(self):
        
        image_names = [self._convert_source_path_to_target_name(p) for p in (self.dataset_root / "Kitti/raw/training/image_2").glob("*.png")]
        image_names.sort()
        random.Random(SEED).shuffle(image_names)
        n = len(image_names)
        
        train_image_names = image_names[:int(n*self.train_ratio)]
        val_image_names = image_names[int(n*self.train_ratio):]

        json.dump({
            "train": train_image_names, 
            "val": val_image_names
        }, open(self.target_dataset_root / "split.json", "w"))

    def prepare_images(self):

        split_file = json.load((self.target_dataset_root / "split.json").open())
        image_names = {}

        def _prepare_images_split(split):
            os.makedirs(self.target_dataset_root / split / "data", exist_ok=True)
            os.chdir(self.target_dataset_root / split / "data")
            
            
            image_names[split] = []
            for p in tqdm(Path(self.dataset_root / "Kitti/raw/training/image_2").glob("*.png"), desc="Creating soft link"):
                tgt_name = self._convert_source_path_to_target_name(p)
                if tgt_name in split_file[split]:
                    
                    cmd = f"ln -s {p} {tgt_name}"
                    if not Path(tgt_name).exists():
                        os.system(cmd)
                    
                    image_names[split].append(tgt_name)

        _prepare_images_split("train")
        _prepare_images_split("val")
        self._image_names = image_names

    def prepare_labels(self):
                
        assert hasattr(self, "_image_names"), "Need to call self.prepare_images first"

        def _prepare_labels_split(split):
            

            os.chdir(self.target_dataset_root / split)

            labels = {
                "info": {},
                "licenses": [],
                "categories": [{
                    "id": i,
                    "name": c,
                    "supercategory": "all"
                } for i, c in enumerate(self.KITTI_CLASSES)],
                "images": [],
                "annotations": []
            }


            def __construct_image_dict():
                image_root = Path(self.target_dataset_root / split / "data")
                
                image_id = 1
                file_name_to_image_info = {}
                for p in self._image_names[split]:

                    w, h = imagesize.get(image_root / p)#Image.open(image_root / p).size
                    img_dict = {
                        "id": image_id,
                        "license": 1,
                        "file_name": p,
                        "height": h,
                        "width": w,
                    }
                    labels["images"].append(img_dict)
                    file_name_to_image_info[p] = (image_id, w, h)
                    image_id += 1
                return file_name_to_image_info

            def __construct_annotations_dict(file_name_to_image_info):

                anno_id = 1
                for anno_p in tqdm((self.dataset_root / "Kitti/raw/training/label_2").glob("*.txt"), desc=f"Process Kitti {split}"):
                    
                    orig_p = anno_p.with_suffix(".png")
                    file_name = self._convert_source_path_to_target_name(orig_p)
                    if file_name in self._image_names[split]:

                        df = pd.read_csv(anno_p, sep=" ", index_col=False, header=None)
                        image_id, image_w, image_h = file_name_to_image_info[file_name]

                        for i, row in df.iterrows():

                            left, top, right, bottom = row[4:8]

                            x_top_left, y_top_left, width, height = left, top, right-left, bottom-top
                            bbox = [x_top_left, y_top_left, width, height]
                            bbox = self.shift_or_scale_bbox(bbox, image_w, image_h)

                            category_id = self.KITTI_CLASSES.index(row[0])

                            new_anno_dict = {
                                "id": anno_id,
                                "image_id": image_id,
                                "bbox": bbox,
                                "category_id": category_id,
                                "iscrowd": 0,
                                "area": width*height
                            }
                            labels["annotations"].append(new_anno_dict)
                            anno_id += 1
                            
            file_name_to_image_info = __construct_image_dict()
            __construct_annotations_dict(file_name_to_image_info)
            return labels


        train_labels = _prepare_labels_split("train")
        json.dump(train_labels, open(self.target_dataset_root / "train" / "labels.json", "w"))

        val_labels = _prepare_labels_split("val")
        json.dump(val_labels, open(self.target_dataset_root / "val" / "labels.json", "w"))


class VirtualKittiPrepare(BaseCOCOPrepare):

    KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]
    VARIANTS = ["clone", "morning", "15-deg-left", "15-deg-right"]#, "sunset", "overcast", "fog", "rain"]

    def __init__(self, shift, scale, train_ratio=0.7, image_prefix=""):

        dataset_name = "virtual_kitti"
        dataset_root = "/datasets/virtual_kitti"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)

        super(VirtualKittiPrepare, self).__init__(dataset_name, dataset_root)

        self.train_ratio = train_ratio
        self.image_prefix = image_prefix

    def _convert_source_path_to_target_name(self, src_p):
        return f"{self.image_prefix}{'_'.join(src_p.parts[-3:])}"

    def prepare_splits_if_needed(self):

        image_names = []
        for variant in self.VARIANTS:
            for p in self.dataset_root.glob(f"vkitti_1.3.1_rgb/*/{variant}/*.png"):

                tgt_name = self._convert_source_path_to_target_name(p)
                image_names.append(tgt_name)

        image_names.sort()
        random.Random(SEED).shuffle(image_names)
        n = len(image_names)
        
        train_image_names = image_names[:int(n*self.train_ratio)]
        val_image_names = image_names[int(n*self.train_ratio):]

        json.dump({
            "train": train_image_names, 
            "val": val_image_names
        }, open(self.target_dataset_root / "split.json", "w"))
        
    def prepare_images(self):

        split_file = json.load((self.target_dataset_root / "split.json").open())

        image_names = {}

        def _prepare_images_split(split):

            os.makedirs(self.target_dataset_root / split / "data", exist_ok=True)
            os.chdir(self.target_dataset_root / split / "data")
            
            image_names[split] = []

            for variant in self.VARIANTS:    
                for p in self.dataset_root.glob(f"vkitti_1.3.1_rgb/*/{variant}/*.png"):

                    tgt_name = self._convert_source_path_to_target_name(p)
                    if tgt_name in split_file[split]:
                        cmd = f"ln -s {p} {tgt_name}"
                        if not Path(tgt_name).exists():
                            os.system(cmd)

                        image_names[split].append(tgt_name)

        _prepare_images_split("train")
        _prepare_images_split("val")
        self._image_names = image_names

    def prepare_labels(self):

        assert hasattr(self, "_image_names"), "Need to call self.prepare_images first"


        def _prepare_labels_split(split):
    
            os.chdir(self.target_dataset_root / split)
                
            labels = {
                "info": {},
                "licenses": [],
                "categories": [{
                    "id": i,
                    "name": c,
                    "supercategory": "all"
                } for i, c in enumerate(self.KITTI_CLASSES)],
                "images": [],
                "annotations": []
            }


            def __construct_image_dict():
                image_root = Path(self.target_dataset_root / split / "data")
                image_id = 1
                file_name_to_image_info = {}
            
                for p in self._image_names[split]:

                    w, h = imagesize.get(image_root / p)#Image.open(image_root / p).size
                    img_dict = {
                        "id": image_id,
                        "license": 1,
                        "file_name": p,
                        "height": h,
                        "width": w,
                    }
                    labels["images"].append(img_dict)
                    file_name_to_image_info[p] = (image_id, w, h)
                    image_id += 1

                return file_name_to_image_info

            def __construct_annotations_dict(file_name_to_image_info):

                anno_id = 1
                for anno_p in tqdm(self.dataset_root.glob("vkitti_1.3.1_motgt/*.txt"), desc=f"Process Virtual Kitti {split}"):
                    if any([v in str(anno_p) for v in self.VARIANTS]):

                            world, variation = anno_p.stem.split("_")

                            df = pd.read_csv(anno_p, sep=" ", index_col=False)

                            for i, row in df.iterrows():
                                anno_dict = row.to_dict()
                                frame = anno_dict.pop("frame")

                                file_name = f"{self.image_prefix}{world}_{variation}_{str(frame).zfill(5)}.png"

                                if file_name in self._image_names[split]:
                                        
                                    image_id, image_w, image_h = file_name_to_image_info[file_name]

                                    left, top, right, bottom = anno_dict["l"], anno_dict["t"], anno_dict["r"], anno_dict["b"]
                                    x_top_left, y_top_left, width, height = left, top, right-left, bottom-top
                                    
                                    bbox = [x_top_left, y_top_left, width, height]

                                    if self.shift != "no":
                                        bbox = self.shift_bbox(bbox, image_w, image_h)

                                    if self.scale != "no":
                                        bbox = self.scale_bbox(bbox, image_w, image_h)

                                    category_id = self.KITTI_CLASSES.index(anno_dict["label"])


                                    new_anno_dict = {
                                        "id": anno_id,
                                        "image_id": image_id,
                                        "bbox": bbox,
                                        "category_id": category_id,
                                        "iscrowd": 0,
                                        "area": width*height
                                    }
                                    labels["annotations"].append(new_anno_dict)
                                    anno_id += 1

            file_name_to_image_info = __construct_image_dict()
            __construct_annotations_dict(file_name_to_image_info)
            return labels


        train_labels = _prepare_labels_split("train")
        json.dump(train_labels, open(self.target_dataset_root / "train" / "labels.json", "w"))

        val_labels = _prepare_labels_split("val")
        json.dump(val_labels, open(self.target_dataset_root / "val" / "labels.json", "w"))


class SynscapesPrepare(BaseCOCOPrepare):

    def __init__(self, shift, scale, train_ratio=0.7, image_prefix=""):

        dataset_name = "synscapes"
        dataset_root = "/datasets/synscapes"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)

        super(SynscapesPrepare, self).__init__(dataset_name, dataset_root)
        
        self.train_ratio = train_ratio
        self.image_prefix = image_prefix

    def _convert_source_path_to_target_name(self, src_p):
        return f"{self.image_prefix}{src_p.name}"

    def prepare_splits_if_needed(self):
        
        image_names = [self._convert_source_path_to_target_name(p) for p in (self.dataset_root / "img/rgb").glob("*.png")]
        image_names.sort()
        random.Random(SEED).shuffle(image_names)
        n = len(image_names)
        
        train_image_names = image_names[:int(n*self.train_ratio)]
        val_image_names = image_names[int(n*self.train_ratio):]

        json.dump({
            "train": train_image_names, 
            "val": val_image_names
        }, open(self.target_dataset_root / "split.json", "w"))
    
    def prepare_images(self):

        split_file = json.load((self.target_dataset_root / "split.json").open())
        image_names = {}

        def _prepare_images_split(split):
            os.makedirs(self.target_dataset_root / split / "data", exist_ok=True)
            os.chdir(self.target_dataset_root / split / "data")

            image_names[split] = []
            for p in tqdm(Path(self.dataset_root / "img/rgb").glob("*.png"), desc="Creating soft link"):
                if p.name.startswith("."):
                    continue

                tgt_name = self._convert_source_path_to_target_name(p)
                if tgt_name in split_file[split]:
                    
                    cmd = f"ln -s {p} {tgt_name}"
                    if not Path(tgt_name).exists():
                        os.system(cmd)
                    
                    image_names[split].append(tgt_name)

        _prepare_images_split("train")
        _prepare_images_split("val")
        self._image_names = image_names

    def prepare_labels(self):
        assert hasattr(self, "_image_names"), "Need to call self.prepare_images first"
        
        def _prepare_labels_split(split):
            os.chdir(self.target_dataset_root / split)
            
            # Use Cityscape id
            labels = {
                "info": {},
                "licenses": [],
                "categories": [{
                    "id": l.id,
                    "name": l.name,
                    "supercategory": "all"
                } for l in CITYSCAPES_LABELS if l.id  != -1],
                "images": [],
                "annotations": []
            }

            def __construct_image_dict():

                image_root = Path(self.target_dataset_root / split / "data")
                
                image_id = 1
                file_name_to_image_info = {}
                for p in self._image_names[split]:

                    w, h = imagesize.get(image_root / p)#Image.open(image_root / p).size
                    img_dict = {
                        "id": image_id,
                        "license": 1,
                        "file_name": p,
                        "height": h,
                        "width": w,
                    }
                    labels["images"].append(img_dict)
                    file_name_to_image_info[p] = (image_id, w, h)
                    image_id += 1
                return file_name_to_image_info

            def __construct_annotations_dict(file_name_to_image_info):
                
                anno_id = 1
                for anno_p in tqdm(Path(self.dataset_root / "meta").glob("*.json"), desc=f"Process synscape {split}"):
    
                    #orig_p = anno_p.with_suffix(".png")

                    orig_p = anno_p.parent.parent / "img" / "rgb" / anno_p.with_suffix(".png").name
                    file_name = self._convert_source_path_to_target_name(orig_p)

                    if file_name in self._image_names[split]:

                        image_id, image_w, image_h = file_name_to_image_info[file_name]

                        metadata = json.load(open(anno_p))
                        bbox2d_dict = metadata["instance"]["bbox2d"]
                        class_dict = metadata["instance"]["class"]
                        occluded_dict = metadata["instance"]["occluded"]
                        truncated_dict = metadata["instance"]["truncated"]

                        for instance_id in bbox2d_dict.keys():

                            if class_dict[instance_id] != -1:
                                
                                x_top_left = bbox2d_dict[instance_id]["xmin"] * image_w
                                y_top_left = bbox2d_dict[instance_id]["ymin"] * image_h
                                width = (bbox2d_dict[instance_id]["xmax"] - bbox2d_dict[instance_id]["xmin"]) * image_w
                                height = (bbox2d_dict[instance_id]["ymax"] - bbox2d_dict[instance_id]["ymin"]) * image_h
                                
                                bbox = [x_top_left, y_top_left, width, height]
                                bbox = self.shift_or_scale_bbox(bbox, image_w, image_h)
                                
                                new_anno_dict = {
                                    "id": anno_id, 
                                    "image_id": image_id, 
                                    "bbox": bbox, 
                                    "category_id": class_dict[instance_id], 
                                    "occluded": occluded_dict[instance_id], 
                                    "truncated": truncated_dict[instance_id], 
                                    "iscrowd": 0, 
                                    "area": width*height
                                }

                                labels["annotations"].append(new_anno_dict)
                                anno_id += 1

            file_name_to_image_info = __construct_image_dict()
            __construct_annotations_dict(file_name_to_image_info)
            return labels

        train_labels = _prepare_labels_split("train")
        json.dump(train_labels, open(self.target_dataset_root / "train" / "labels.json", "w"))

        val_labels = _prepare_labels_split("val")
        json.dump(val_labels, open(self.target_dataset_root / "val" / "labels.json", "w"))

        
class CityscapesPrepare(BaseCOCOPrepare):

    def __init__(self, shift, scale, image_prefix=""):

        dataset_name = "cityscapes"
        dataset_root = "/datasets/cityscapes"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)

        super(CityscapesPrepare, self).__init__(dataset_name, dataset_root)

        self.image_prefix = image_prefix

    def _convert_source_path_to_target_name(self, src_p):
        return f"{self.image_prefix}{src_p.name}"

    def prepare_splits_if_needed(self):
        pass

    def prepare_images(self):
        
        image_names = {
            "train": [], 
            "val": []
        }

        def _prepare_images_split(split):
            os.makedirs(self.target_dataset_root / f"{split}/data", exist_ok=True)
            os.chdir(self.target_dataset_root / f"{split}/data")

            for p in Path(self.dataset_root / f"leftImg8bit/{split}").glob(f"*/*.png"):

                tgt_name = p.name
                cmd = f"ln -s {p} {tgt_name}"
                if not Path(tgt_name).exists():
                    os.system(cmd)
                image_names[split].append(tgt_name)


        _prepare_images_split("train")
        _prepare_images_split("val")
        self._image_names = image_names

    def prepare_labels(self):
        
        from cityscapesscripts.helpers.annotation import CsBbox3d
        from cityscapesscripts.helpers.box3dImageTransform import (
            Camera, 
            Box3dImageTransform,
            CRS_V,
            CRS_C,
            CRS_S
        )
        
        assert hasattr(self, "_image_names"), "Need to call self.prepare_images first"
        
        def _prepare_labels_split(split):

            os.chdir(self.target_dataset_root / split)
            
            labels = {
                "info": {},
                "licenses": [],
                "categories": [{
                    "id": l.id,
                    "name": l.name,
                    "supercategory": "all"
                } for l in CITYSCAPES_LABELS if l.id  != -1],
                "images": [],
                "annotations": []
            }
            
            class_name_to_id = {i["name"]: i["id"] for i in labels["categories"]}


            def __construct_image_dict():
                image_root = Path(self.target_dataset_root / f"{split}/data")
                image_id = 1
                file_name_to_image_info= {}

                for p in self._image_names[split]:

                    w, h = imagesize.get(image_root / p)#Image.open(image_root / p).size
                    img_dict = {
                        "id": image_id,
                        "license": 1,
                        "file_name": p,
                        "height": h,
                        "width": w,
                    }
                    labels["images"].append(img_dict)
                    file_name_to_image_info[p] = (image_id, w, h)
                    image_id += 1

                return file_name_to_image_info

            def __construct_annotations_dict(file_name_to_image_info):
                anno_id = 1
                for anno_p in tqdm(Path(self.dataset_root / f"gtBbox3d/{split}").glob("*/*.json"), desc=f"Processing cityscape {split}"):


                    file_name = anno_p.with_suffix(".png").name.replace("gtBbox3d", "leftImg8bit")
                    if file_name in self._image_names[split]:
                        metadata = json.load(open(anno_p))
                        
                        camera = Camera(fx=metadata["sensor"]["fx"],
                                fy=metadata["sensor"]["fy"],
                                u0=metadata["sensor"]["u0"],
                                v0=metadata["sensor"]["v0"],
                                sensor_T_ISO_8855=metadata["sensor"]["sensor_T_ISO_8855"]
                        )

                        box3d_annotation = Box3dImageTransform(camera=camera)    
                        objects_dict = metadata["objects"]
                            
                        
                        image_id, image_w, image_h = file_name_to_image_info[file_name]
                        
                        for obj_dict in objects_dict:
                            obj = CsBbox3d()
                            obj.fromJsonText(obj_dict)

                            
                            x_top_left, y_top_left, x_bottom_right, y_bottom_right = obj.bbox_2d.bbox_modal
                            bbox_width = x_bottom_right - x_top_left
                            bbox_height = y_bottom_right - y_top_left
                            
                            bbox = [x_top_left, y_top_left, bbox_width, bbox_height]
                            bbox = self.shift_or_scale_bbox(bbox, image_w, image_h)
                            
                            new_anno_dict = {
                                "id": anno_id, 
                                "image_id": image_id, 
                                "bbox": bbox, 
                                "category_id": class_name_to_id[obj_dict["label"]], 
                                "occluded": obj_dict["occlusion"], 
                                "truncated": obj_dict["truncation"], 
                                "iscrowd": 0, 
                                "area": bbox_width*bbox_height
                            }

                            labels["annotations"].append(new_anno_dict)
                            anno_id += 1
                    
            file_name_to_image_info = __construct_image_dict()
            __construct_annotations_dict(file_name_to_image_info)
            return labels
                    

        train_labels = _prepare_labels_split("train")
        json.dump(train_labels, open(self.target_dataset_root / "train" / "labels.json", "w"))

        val_labels = _prepare_labels_split("val")
        json.dump(val_labels, open(self.target_dataset_root / "val" / "labels.json", "w"))


class MixedDatasetsPrepare(BaseCOCOPrepare):
    """Mix two datasets. The train and val splits follow their original splits
    """

    def __init__(self, datasetA, datasetB):

        dataset_name = f"mixed_{datasetA.dataset_name}_{datasetB.dataset_name}"
        super(MixedDatasetsPrepare, self).__init__(dataset_name, None)

        self.datasetA = datasetA
        self.datasetB = datasetB
        
        print(f"Prepare {datasetA.dataset_name}")
        self.datasetA.prepare()
        print(f"Prepare {datasetB.dataset_name}")
        self.datasetB.prepare()

    def prepare_splits_if_needed(self):

        splits = {
            "train": [], 
            "val": []
        }

        def _prepare_split(split):
            for p in (self.datasetA.target_dataset_root / split / "data").glob("*.png"):
                splits[split].append(f"{p.name}")

            for p in (self.datasetB.target_dataset_root / split / "data").glob("*.png"):
                splits[split].append(p.name)

        _prepare_split("train")
        _prepare_split("val")
        json.dump(splits, open(self.target_dataset_root / "split.json", "w"))
        
    def prepare_images(self):

        split_file = json.load((self.target_dataset_root / "split.json").open())

        image_names = {}

        def _prepare_images_split(split):

            os.makedirs(self.target_dataset_root / split / "data", exist_ok=True)
            os.chdir(self.target_dataset_root / split / "data")
            
            image_names[split] = []

            for p in (self.datasetA.target_dataset_root / split / "data").glob("*.png"):
                tgt_name = f"{self.datasetA.dataset_name}-{p.name}"
                cmd = f"ln -s {p} {tgt_name}"
                if not Path(tgt_name).exists():
                    os.system(cmd)
                image_names[split].append(tgt_name)
            
            for p in (self.datasetB.target_dataset_root / split / "data").glob("*.png"):
                tgt_name = f"{self.datasetB.dataset_name}-{p.name}"
                cmd = f"ln -s {p} {tgt_name}"
                if not Path(tgt_name).exists():
                    os.system(cmd)
                image_names[split].append(tgt_name)
                
        _prepare_images_split("train")
        _prepare_images_split("val")
        self._image_names = image_names

    def prepare_labels(self):

        def _prepare_labels_split(split):
            
            os.chdir(self.target_dataset_root / split)

            
            datasetA_labels = json.load(open(self.datasetA.target_dataset_root / split / "labels.json"))
            labels = copy.deepcopy(datasetA_labels)
            for img_dict in labels["images"]:
                name = img_dict["file_name"]
                img_dict["file_name"] = f"{self.datasetA.dataset_name}-{name}"

            # +1 to ensure no overlap
            image_id_offset = max([i["id"] for i in labels["images"]]) + 1

            datasetB_labels = json.load(open(self.datasetB.target_dataset_root / split / "labels.json"))
            for img_dict in datasetB_labels["images"]:
                name = img_dict["file_name"]
                img_dict["file_name"] = f"{self.datasetB.dataset_name}-{name}"
                img_dict["id"] = image_id_offset + img_dict["id"]

                labels["images"].append(img_dict)

            # +1 to ensure no overlap
            anno_id_offset = max([i["id"] for i in labels["annotations"]]) + 1
            for anno_dict in datasetB_labels["annotations"]:
                anno_dict["image_id"] = image_id_offset + anno_dict["image_id"]
                anno_dict["id"] = anno_id_offset + anno_dict["id"]
                labels["annotations"].append(anno_dict)

            return labels

        train_labels = _prepare_labels_split("train")
        json.dump(train_labels, open(self.target_dataset_root / "train" / "labels.json", "w"))

        val_labels = _prepare_labels_split("val")
        json.dump(val_labels, open(self.target_dataset_root / "val" / "labels.json", "w"))

