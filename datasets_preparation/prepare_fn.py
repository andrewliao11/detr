import os

from pathlib import Path
import ipdb

class BaseCOCOPrepare():
    def __init__(self, dataset_name, dataset_root):
        self.dataset_name = dataset_name
        self.dataset_root = dataset_root
        self.target_dataset_root = os.environ["HOME"] / f"datasets/{self.dataset_name}/coco_format"

        
    def prepare_images(self):
        raise NotImplementedError


    def prepare(self):
        self.prepare_images()
        self.prepare_labels()


class Mscoco14Prepare(BaseCOCOPrepare):
    def __init__(self):
        super(Mscoco14Prepare, self).__init__("mscoco14", "/datasets/mscoco14")

    def prepare_images(self):

        os.makedirs(self.target_dataset_root / "train", exist_ok=True)
        os.chdir(self.target_dataset_root / "train")
        cmd = f"ln -s {self.dataset_root / 'train2014/train2014'} data"
        os.system(cmd)

        os.makedirs(target_dataset_root / "val", exist_ok=True)
        os.chdir(target_dataset_root / "val")
        cmd = f"ln -s {dataset_root / 'val2014/val2014'} data"
        os.system(cmd)

    def prepare_labels(self):
        os.chdir(self.target_dataset_root / "train")
        cmd = f"ln -s {dataset_root / 'instances_train-val2014/annotations/instances_train2014.json'} labels.json"
        os.system(cmd)

        os.chdir(target_dataset_root / "val")
        cmd = f"ln -s {dataset_root / 'instances_train-val2014/annotations/instances_val2014.json'} labels.json"
        os.system(cmd)


class Mscoco17Prepare(BaseCOCOPrepare):
    def __init__(self):
        super(Mscoco17Prepare, self).__init__("mscoco17", "/datasets/mscoco17")

    def prepare_images(self):

        os.makedirs(target_dataset_root / "train", exist_ok=True)
        os.chdir(target_dataset_root / "train")
        cmd = f"ln -s {dataset_root / 'train2017'} data"
        os.system(cmd)

        os.makedirs(target_dataset_root / "val", exist_ok=True)
        os.chdir(target_dataset_root / "val")
        cmd = f"ln -s {dataset_root / 'val2017'} data"
        os.system(cmd)

    def prepare_labels(self):

        os.chdir(target_dataset_root / "train")
        cmd = f"ln -s {dataset_root / 'annotations/instances_train2017.json'} labels.json"
        os.system(cmd)

        os.chdir(target_dataset_root / "val")

        cmd = f"ln -s {dataset_root / 'annotations/instances_val2017.json'} labels.json"
        os.system(cmd)


class KittiPrepare(BaseCOCOPrepare):
    KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]

    def __init__(self):
        super(KittiPrepare, self).__init__("kitti", "/datasets/kitti")

    def prepare_images(self):

        os.makedirs(self.target_dataset_root / "data", exist_ok=True)

        os.chdir(self.target_dataset_root / "data")        
        image_paths = []
        for p in Path(self.dataset_root / "Kitti/raw/training/image_2").glob(f"*.png"):

            tgt_name = p.name
            cmd = f"ln -s {p} {tgt_name}"
            if not Path(tgt_name).exists():
                os.system(cmd)
            image_paths.append(tgt_name)

        self._image_paths = image_paths

    def prepare_labels(self):
                
        assert hasattr(self, "_image_paths"), "Need to call self.prepare_images first"

        os.chdir(target_dataset_root)

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

        image_root = Path(self.target_dataset_root / "data")
        image_id = 1
        file_name_to_image_id = {}
        

        for p in self._image_paths:

            w, h = Image.open(image_root / p).size
            img_dict = {
                "id": image_id,
                "license": 1,
                "file_name": p,
                "height": h,
                "width": w,
            }
            labels["images"].append(img_dict)
            file_name_to_image_id[p] = image_id
            image_id += 1


        anno_id = 1
        for anno_p in Path(self.dataset_root / "Kitti/raw/training/label_2").glob("*.txt"):
            
            
            df = pd.read_csv(anno_p, sep=" ", index_col=False, header=None)
            file_name = anno_p.with_suffix(".png").name
            image_id = file_name_to_image_id[file_name]


            for i, row in df.iterrows():
                
                left, top, right, bottom = row[4:8]
                x_top_left, y_top_left, width, height = left, top, right-left, bottom-top
                category_id = self.KITTI_CLASSES.index(row[0])

                new_anno_dict = {
                    "id": anno_id,
                    "image_id": image_id,
                    "bbox": [x_top_left, y_top_left, width, height],
                    "category_id": category_id,
                    "iscrowd": 0,
                    "area": width*height
                }
                labels["annotations"].append(new_anno_dict)
                anno_id += 1


        json.dump(labels, open("labels.json", "w"))


class VirtualKittiPrepare(BaseCOCOPrepare):

    KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]
    VARIANTS = ["clone", "morning", "sunset", "overcast", "fog", "rain"]

    def __init__(self):
        super(VirtualKittiPrepare, self).__init__("virtual_kitti", "/datasets/virtual_kitti")

    def prepare_images(self):

        os.makedirs(self.target_dataset_root / "data", exist_ok=True)
        os.chdir(self.target_dataset_root / "data")

        image_paths = []
        for variant in self.VARIANTS:
            for p in self.dataset_root.glob(f"vkitti_1.3.1_rgb/*/{variant}/*.png"):

                tgt_name = "_".join(p.parts[-3:])
                cmd = f"ln -s {p} {tgt_name}"
                if not Path(tgt_name).exists():
                    os.system(cmd)
                image_paths.append(tgt_name)

        self._image_paths = image_paths

        os.chdir(self.target_dataset_root)

    def prepare_labels(self):

        assert hasattr(self, "_image_paths"), "Need to call self.prepare_images first"

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


        image_root = Path(self.target_dataset_root / "data")
        image_id = 1
        file_name_to_image_id = {}
        for p in image_paths:

            w, h = Image.open(image_root / p).size
            img_dict = {
                "id": image_id,
                "license": 1,
                "file_name": p,
                "height": h,
                "width": w,
            }
            labels["images"].append(img_dict)
            file_name_to_image_id[p] = image_id
            image_id += 1



        anno_id = 1
        for anno_p in self.dataset_root.glob("vkitti_1.3.1_motgt/*.txt"):
            if any([v in str(anno_p) for v in self.VARIANTS]):

                    world, variation = anno_p.stem.split("_")

                    df = pd.read_csv(anno_p, sep=" ", index_col=False)

                    for i, row in df.iterrows():
                        anno_dict = row.to_dict()
                        frame = anno_dict.pop("frame")

                        file_name = f"{world}_{variation}_{str(frame).zfill(5)}.png"
                        image_id = file_name_to_image_id[file_name]

                        left, top, right, bottom = anno_dict["l"], anno_dict["t"], anno_dict["r"], anno_dict["b"]
                        x_top_left, y_top_left, width, height = left, top, right-left, bottom-top
                        category_id = self.KITTI_CLASSES.index(anno_dict["label"])


                        new_anno_dict = {
                            "id": anno_id,
                            "image_id": image_id,
                            "bbox": [x_top_left, y_top_left, width, height],
                            "category_id": category_id,
                            "iscrowd": 0,
                            "area": width*height
                        }
                        labels["annotations"].append(new_anno_dict)
                        anno_id += 1


        json.dump(labels, open("labels.json", "w"))



class SynscapesPrepare(BaseCOCOPrepare):

    def __init__(self):
        super(SynscapesPrepare, self).__init__("synscapes", "/datasets/synscapes")

    def prepare_images(self):

        os.makedirs(self.target_dataset_root / "data", exist_ok=True)
        os.chdir(self.target_dataset_root / "data")

        image_paths = []
        for p in Path(self.dataset_root / "img/rgb").glob(f"*.png"):

            if p.name.startswith("."):
                continue

            tgt_name = p.name
            cmd = f"ln -s {p} {tgt_name}"
            if not Path(tgt_name).exists():
                os.system(cmd)
            image_paths.append(tgt_name)

        self._image_paths = image_paths

    def prepare_labels(self):
        
        assert hasattr(self, "_image_paths"), "Need to call self.prepare_images first"

        os.chdir(self.target_dataset_root)

        # https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
        from cityscape_labels import labels as LABELS
        
        # Use Cityscape id
        labels = {
            "info": {},
            "licenses": [],
            "categories": [{
                "id": l.id,
                "name": l.name,
                "supercategory": "all"
            } for l in LABELS if l.id  != -1],
            "images": [],
            "annotations": []
        }

        image_root = Path(self.target_dataset_root / "data")
        image_id = 1
        file_name_to_image_info= {}

        for p in self._image_paths:

            if p.startswith("."):
                continue

            w, h = Image.open(image_root / p).size
            img_dict = {
                "id": image_id,
                "license": 1,
                "file_name": p,
                "height": h,
                "width": w,
            }
            labels["images"].append(img_dict)
            file_name_to_image_info[p] = (image_id, h, w)
            image_id += 1



        anno_id = 1
        for anno_p in Path(self.dataset_root / "meta").glob("*.json"):
            
            metadata = json.load(open(anno_p))
            bbox2d_dict = metadata["instance"]["bbox2d"]
            class_dict = metadata["instance"]["class"]
            occluded_dict = metadata["instance"]["occluded"]
            truncated_dict = metadata["instance"]["truncated"]

            file_name = anno_p.with_suffix(".png").name 
            image_id, image_height, image_width = file_name_to_image_info[file_name]
            for instance_id in bbox2d_dict.keys():

                if class_dict[instance_id] != -1:
                    x_top_left = bbox2d_dict[instance_id]["xmin"] * image_width
                    y_top_left = bbox2d_dict[instance_id]["ymin"] * image_height
                    width = (bbox2d_dict[instance_id]["xmax"] - bbox2d_dict[instance_id]["xmin"]) * image_width
                    height = (bbox2d_dict[instance_id]["ymax"] - bbox2d_dict[instance_id]["ymin"]) * image_height
                    new_anno_dict = {
                        "id": anno_id, 
                        "image_id": image_id, 
                        "bbox": [x_top_left, y_top_left, width, height], 
                        "category_id": class_dict[instance_id], 
                        "occluded": occluded_dict[instance_id], 
                        "truncated": truncated_dict[instance_id], 
                        "iscrowd": 0, 
                        "area": width*height
                    }

                    labels["annotations"].append(new_anno_dict)
                    anno_id += 1


        json.dump(labels, open("labels.json", "w"))
