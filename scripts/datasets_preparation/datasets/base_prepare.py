import os
import json
import random
import imagesize

from pathlib import Path
from collections import Counter
from tqdm import tqdm

SEED = 42


class BasePrepare():
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


class NonCOCOBasePrepare(BasePrepare):

    def __init__(self, dataset_name, dataset_root, train_ratio, image_prefix):

        super(NonCOCOBasePrepare, self).__init__(dataset_name, dataset_root)
        self.train_ratio = train_ratio
        self.image_prefix = image_prefix

    def _prepare_splits_if_needed(self):
        
        image_names = [self._convert_source_path_to_target_name(p) for p in self._get_images_iterator()]
        image_names.sort()
        random.Random(SEED).shuffle(image_names)
        n = len(image_names)
        
        train_image_names = image_names[:int(n*self.train_ratio)]
        val_image_names = image_names[int(n*self.train_ratio):]

        json.dump({
            "train": train_image_names, 
            "val": val_image_names
        }, open(self.target_dataset_root / "split.json", "w"))


    def _prepare_images(self):

        split_file = json.load((self.target_dataset_root / "split.json").open())
        image_names = {}

        def _prepare_images_split(split):
            os.makedirs(self.target_dataset_root / split / "data", exist_ok=True)
            os.chdir(self.target_dataset_root / split / "data")
            
            
            image_names[split] = []
            for p in tqdm(self._get_images_iterator(), desc="Creating soft link"):
                tgt_name = self._convert_source_path_to_target_name(p)
                if tgt_name in split_file[split]:
                    
                    cmd = f"ln -s {p} {tgt_name}"
                    if not Path(tgt_name).exists():
                        os.system(cmd)
                    
                    image_names[split].append(tgt_name)

        _prepare_images_split("train")
        _prepare_images_split("val")
        self._image_names = image_names

    def _prepare_labels(self):

        assert hasattr(self, "_image_names"), "Need to call self.prepare_images first"
        
        def _prepare_labels_split(split):

            os.chdir(self.target_dataset_root / split)
            labels = self.init_labels()

            def __construct_image_dict():
                image_root = Path(self.target_dataset_root / split / "data")
                
                image_id = 1
                filename_to_image_info = {}
                for p in self._image_names[split]:

                    w, h = imagesize.get(image_root / p)
                    img_dict = {
                        "id": image_id,
                        "license": 1,
                        "file_name": p,
                        "height": h,
                        "width": w,
                    }
                    labels["images"].append(img_dict)
                    filename_to_image_info[p] = (image_id, w, h)
                    image_id += 1
                return filename_to_image_info

            def __construct_annotations_dict(filename_to_image_info):

                anno_id = 1

                for anno_p in tqdm(self._get_annotation_iterator(split), desc=f"Process {split}"):
                    new_anno_dicts, anno_id = self._create_anno_dict(anno_p, anno_id, filename_to_image_info, self._image_names[split])
                    labels["annotations"].extend(new_anno_dicts)

                print(Counter([i["category_id"] for i in labels["annotations"]]))

            filename_to_image_info = __construct_image_dict()
            __construct_annotations_dict(filename_to_image_info)
            return labels


        train_labels = _prepare_labels_split("train")
        json.dump(train_labels, open(self.target_dataset_root / "train" / "labels.json", "w"))

        val_labels = _prepare_labels_split("val")
        json.dump(val_labels, open(self.target_dataset_root / "val" / "labels.json", "w"))
        return train_labels, val_labels
