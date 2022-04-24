import os
import pandas as pd
from tqdm import tqdm

from .base_prepare import NonCOCOBasePrepare


class KittiPrepare(NonCOCOBasePrepare):
    KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]

    def __init__(self, shift, scale, train_ratio=0.7, image_prefix=""):

        dataset_name = "kitti"
        dataset_root = "/datasets/kitti"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)
        
        super(KittiPrepare, self).__init__(dataset_name, dataset_root, train_ratio, image_prefix)

    def init_labels(self):
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
        return labels

    def _convert_source_path_to_target_name(self, src_p):
        return f"{self.image_prefix}{src_p.name}"

    def _get_images_iterator(self):
        return (self.dataset_root / "Kitti/raw/training/image_2").glob("*.png")

    def _get_annotation_iterator(self, split):
        return (self.dataset_root / "Kitti/raw/training/label_2").glob("*.txt")

    def _create_anno_dict(self, p, start_anno_id, filename_to_image_info, valid_filenames):
        anno_id = start_anno_id
        new_anno_dicts = []
        
        filename = self._convert_source_path_to_target_name(p.with_suffix(".png"))
        

        if filename in valid_filenames:
            image_id, image_w, image_h = filename_to_image_info[filename]
            
            df = pd.read_csv(p, sep=" ", index_col=False, header=None)
            
            for i, row in df.iterrows():

                category_name = row[0]
                category_id = self.KITTI_CLASSES.index(category_name)

                if category_name in self.KITTI_CLASSES:

                    left, top, right, bottom = row[4:8]

                    x_top_left, y_top_left, width, height = left, top, right-left, bottom-top
                    bbox = [x_top_left, y_top_left, width, height]
                    bbox = self.shift_or_scale_bbox(bbox, image_w, image_h)

                    new_anno_dict = {
                        "id": anno_id,
                        "image_id": image_id,
                        "bbox": bbox,
                        "category_id": category_id,
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


class VirtualKittiPrepare(NonCOCOBasePrepare):

    KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]
    VARIANTS = ["clone", "morning", "15-deg-left", "15-deg-right"]#, "sunset", "overcast", "fog", "rain"]

    def __init__(self, shift, scale, train_ratio=0.7, image_prefix=""):

        dataset_name = "virtual_kitti"
        dataset_root = "/datasets/virtual_kitti"

        self.decode_shift_and_scale_str(shift, scale)
        dataset_name = self.change_dataset_name(dataset_name)

        super(VirtualKittiPrepare, self).__init__(dataset_name, dataset_root, train_ratio, image_prefix)

    def init_labels(self):
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
        return labels

    def _convert_source_path_to_target_name(self, src_p):
        return f"{self.image_prefix}{'_'.join(src_p.parts[-3:])}"

    def _get_images_iterator(self):
        for variant in self.VARIANTS:
            for p in self.dataset_root.glob(f"vkitti_1.3.1_rgb/*/{variant}/*.png"):
                yield p
                
    def _get_annotation_iterator(self, split):
        for p in self.dataset_root.glob("vkitti_1.3.1_motgt/*.txt"):
            if any([v in str(p) for v in self.VARIANTS]):
                yield p

    def _create_anno_dict(self, p, start_anno_id, filename_to_image_info, valid_filenames):
        anno_id = start_anno_id
        new_anno_dicts = []

        world, variation = p.stem.split("_")

        df = pd.read_csv(p, sep=" ", index_col=False)
        for i, row in df.iterrows():
            anno_dict = row.to_dict()
            frame = anno_dict.pop("frame")

            filename = f"{self.image_prefix}{world}_{variation}_{str(frame).zfill(5)}.png"

            if filename in valid_filenames:                    
                image_id, image_w, image_h = filename_to_image_info[filename]

                category_name = anno_dict["label"]
                category_id = self.KITTI_CLASSES.index(category_name)

                if category_name in self.KITTI_CLASSES:

                    left, top, right, bottom = anno_dict["l"], anno_dict["t"], anno_dict["r"], anno_dict["b"]
                    x_top_left, y_top_left, width, height = left, top, right-left, bottom-top
                    bbox = [x_top_left, y_top_left, width, height]
                    bbox = self.shift_or_scale_bbox(bbox, image_w, image_h)

                    new_anno_dict = {
                        "id": anno_id,
                        "image_id": image_id,
                        "bbox": bbox,
                        "category_id": category_id,
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
