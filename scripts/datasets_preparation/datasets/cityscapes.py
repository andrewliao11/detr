import os
import json
from pathlib import Path

from .base_prepare import NonCOCOBasePrepare

# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
from .cityscape_labels import labels as CITYSCAPES_LABELS


class SynscapesCarPrepare(NonCOCOBasePrepare):
    
    # person, car, truck, train, bicycle, bus, motocycle
    #INTERESTING_CLASSES = [24, 26, 27, 31, 33, 28, 32]
    INTERESTING_CLASSES = [26]

    def __init__(self, shift, scale, train_ratio=0.7, image_prefix=""):

        dataset_name = "synscapes_car"
        dataset_root = "/datasets/synscapes"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)

        super(SynscapesCarPrepare, self).__init__(dataset_name, dataset_root, train_ratio, image_prefix)

    def init_labels(self):
        # Use Cityscape id
        labels = {
            "info": {},
            "licenses": [],
            "categories": [{
                "id": l.id,
                "name": l.name,
                "supercategory": "all"
            } for l in CITYSCAPES_LABELS if l.id  != -1 and l.id in self.INTERESTING_CLASSES],
            "images": [],
            "annotations": []
        }
        return labels

    def _convert_source_path_to_target_name(self, src_p):
        return f"{self.image_prefix}{src_p.name}"

    def _get_images_iterator(self):
        return (self.dataset_root / "img/rgb").glob("*.png")

    def _get_annotation_iterator(self, split):
        return (self.dataset_root / "meta").glob("*.json")

    def _create_anno_dict(self, p, start_anno_id, filename_to_image_info, valid_filenames):
        anno_id = start_anno_id
        new_anno_dicts = []
        
        image_p = p.parent.parent / "img" / "rgb" / p.with_suffix(".png").name
        filename = self._convert_source_path_to_target_name(image_p)

        if filename in valid_filenames:

            image_id, image_w, image_h = filename_to_image_info[filename]

            metadata = json.load(open(p))
            bbox2d_dict = metadata["instance"]["bbox2d"]
            class_dict = metadata["instance"]["class"]
            occluded_dict = metadata["instance"]["occluded"]
            truncated_dict = metadata["instance"]["truncated"]

            for instance_id in bbox2d_dict.keys():
                
                category_id = class_dict[instance_id]
                if category_id != -1 and category_id in self.INTERESTING_CLASSES:
                    
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
                        "category_id": category_id, 
                        "occluded": occluded_dict[instance_id], 
                        "truncated": truncated_dict[instance_id], 
                        "iscrowd": 0, 
                        "area": width*height
                    }

                    new_anno_dicts.append(new_anno_dict)
                    anno_id += 1

        return new_anno_dicts, anno_id

    def prepare_splits_if_needed(self):
        self._prepare_splits_if_needed()

    def prepare_images(self):
        self._prepare_images()

    def prepare_labels(self):
        self._prepare_labels()


class CityscapesPersonPrepare(NonCOCOBasePrepare):

    # person, car, truck, train, bicycle, bus, motocycle
    INTERESTING_CLASSES = [24]

    def __init__(self, shift, scale, image_prefix=""):

        dataset_name = "cityscapes_person"
        dataset_root = "/datasets/cityscapes"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)

        super(CityscapesPersonPrepare, self).__init__(dataset_name, dataset_root, None, image_prefix)

    def init_labels(self):
        # Use Cityscape id
        labels = {
            "info": {},
            "licenses": [],
            "categories": [{
                "id": l.id,
                "name": l.name,
                "supercategory": "all"
            } for l in CITYSCAPES_LABELS if l.id  != -1 and l.id in self.INTERESTING_CLASSES],
            "images": [],
            "annotations": []
        }
        return labels

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

            for p in (self.dataset_root / f"leftImg8bit/{split}").glob(f"*/*.png"):

                tgt_name = p.name
                cmd = f"ln -s {p} {tgt_name}"
                if not Path(tgt_name).exists():
                    os.system(cmd)
                image_names[split].append(tgt_name)

        _prepare_images_split("train")
        _prepare_images_split("val")
        self._image_names = image_names

    def _get_annotation_iterator(self, split):
        return (self.dataset_root / f"gtBboxCityPersons/{split}").glob("*/*.json")

    def _create_anno_dict(self, p, start_anno_id, filename_to_image_info, valid_filenames):
        anno_id = start_anno_id
        new_anno_dicts = []

        filename = p.with_suffix(".png").name.replace("gtBboxCityPersons", "leftImg8bit")
        if filename in valid_filenames:
            metadata = json.load(open(p))
            image_id, image_w, image_h = filename_to_image_info[filename]
            objects_dict = metadata["objects"]

            for obj_dict in objects_dict:

                category_id = 24
                if category_id in self.INTERESTING_CLASSES:

                    x_top_left, y_top_left, bbox_width, bbox_height = obj_dict["bbox"]
                    
                    bbox = [x_top_left, y_top_left, bbox_width, bbox_height]
                    bbox = self.shift_or_scale_bbox(bbox, image_w, image_h)
                    
                    new_anno_dict = {
                        "id": anno_id, 
                        "image_id": image_id, 
                        "bbox": bbox, 
                        "category_id": category_id, 
                        "occluded": 0., 
                        "truncated": 0.,        # Ignored
                        "iscrowd": 0,           # Ignored
                        "area": bbox_width*bbox_height
                    }

                    new_anno_dicts.append(new_anno_dict)
                    anno_id += 1

        return new_anno_dicts, anno_id

    def prepare_labels(self):
        self._prepare_labels()


class CityscapesCarPrepare(CityscapesPersonPrepare, NonCOCOBasePrepare):

    INTERESTING_CLASSES = [26]
    def __init__(self, shift, scale, image_prefix=""):

        dataset_name = "cityscapes_car"
        dataset_root = "/datasets/cityscapes"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)

        NonCOCOBasePrepare.__init__(self, dataset_name, dataset_root, None, image_prefix)

    def _get_annotation_iterator(self, split):
        return (self.dataset_root / f"gtBbox3d/{split}").glob("*/*.json")

    def _create_anno_dict(self, p, start_anno_id, filename_to_image_info, valid_filenames):
        
        from cityscapesscripts.helpers.annotation import CsBbox3d
        from cityscapesscripts.helpers.box3dImageTransform import (
            Camera, 
            Box3dImageTransform,
            CRS_V,
            CRS_C,
            CRS_S
        )

        init_labels = self.init_labels()
        class_name_to_id = {i["name"]: i["id"] for i in init_labels["categories"]}


        anno_id = start_anno_id
        new_anno_dicts = []


        filename = p.with_suffix(".png").name.replace("gtBbox3d", "leftImg8bit")

        if filename in valid_filenames:
            metadata = json.load(open(p))
            
            camera = Camera(fx=metadata["sensor"]["fx"],
                    fy=metadata["sensor"]["fy"],
                    u0=metadata["sensor"]["u0"],
                    v0=metadata["sensor"]["v0"],
                    sensor_T_ISO_8855=metadata["sensor"]["sensor_T_ISO_8855"]
            )

            box3d_annotation = Box3dImageTransform(camera=camera)    
            objects_dict = metadata["objects"]
                
            
            image_id, image_w, image_h = filename_to_image_info[filename]
            
            for obj_dict in objects_dict:
                obj = CsBbox3d()
                obj.fromJsonText(obj_dict)

                
                if obj_dict["label"] in class_name_to_id and class_name_to_id[obj_dict["label"]] in self.INTERESTING_CLASSES:
                    category_id = class_name_to_id[obj_dict["label"]]

                    x_top_left, y_top_left, x_bottom_right, y_bottom_right = obj.bbox_2d.bbox_modal
                    bbox_width = x_bottom_right - x_top_left
                    bbox_height = y_bottom_right - y_top_left
                    
                    bbox = [x_top_left, y_top_left, bbox_width, bbox_height]
                    bbox = self.shift_or_scale_bbox(bbox, image_w, image_h)
                    
                    new_anno_dict = {
                        "id": anno_id, 
                        "image_id": image_id, 
                        "bbox": bbox, 
                        "category_id": category_id, 
                        "occluded": obj_dict["occlusion"], 
                        "truncated": obj_dict["truncation"], 
                        "iscrowd": 0, 
                        "area": bbox_width*bbox_height
                    }

                    new_anno_dicts.append(new_anno_dict)
                    anno_id += 1
        
        return new_anno_dicts, anno_id

    def prepare_labels(self):
        self._prepare_labels()
