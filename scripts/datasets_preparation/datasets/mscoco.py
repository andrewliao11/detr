import os
from pathlib import Path

from .base_prepare import BasePrepare


class MscocoPrepare(BasePrepare):
    def __init__(self, dataset_name, dataset_root):
        super(MscocoPrepare, self).__init__(dataset_name, dataset_root)
    
    def _prepare_mscoco_images(self, train_images_dir, val_images_dir):
        
        os.makedirs(self.target_dataset_root / "train", exist_ok=True)
        os.chdir(self.target_dataset_root / "train")
        cmd = f"ln -s {train_images_dir} data"
        
        if not Path("data").exists():
            os.system(cmd)

        os.makedirs(self.target_dataset_root / "val", exist_ok=True)
        os.chdir(self.target_dataset_root / "val")
        cmd = f"ln -s {val_images_dir} data"
        
        if not Path("data").exists():
            os.system(cmd)

    def _prepare_labels(self, train_labels_path, val_labels_path):
        os.chdir(self.target_dataset_root / "train")
        cmd = f"ln -s {train_labels_path} labels.json"
        
        if not Path("labels.json").exists():
            os.system(cmd)

        os.chdir(self.target_dataset_root / "val")
        cmd = f"ln -s {val_labels_path} labels.json"
        if not Path("labels.json").exists():
            os.system(cmd)


class Mscoco14Prepare(MscocoPrepare):
    def __init__(self):
        dataset_name = "mscoco14"
        dataset_root = "/datasets/mscoco14"
        super(Mscoco14Prepare, self).__init__(dataset_name, dataset_root)
    
    def prepare_images(self):
        self._prepare_mscoco_images(
            self.dataset_root / "train2014/train2014", 
            self.dataset_root / "val2014/val2014", 
        )

    def prepare_labels(self):
        self._prepare_labels(
            self.dataset_root / "instances_train-val2014/annotations/instances_train2014.json", 
            self.dataset_root / "instances_train-val2014/annotations/instances_val2014.json"
        )


class Mscoco17Prepare(MscocoPrepare):
    def __init__(self):
        dataset_name = "mscoco17"
        dataset_root = "/datasets/mscoco17"
        super(Mscoco17Prepare, self).__init__(dataset_name, dataset_root)

    def prepare_images(self):

        self._prepare_mscoco_images(
            self.dataset_root / "train2017", 
            self.dataset_root / "val2017", 
        )
    
    def prepare_labels(self):
        self._prepare_labels(
            self.dataset_root / "annotations/instances_train2017.json", 
            self.dataset_root / "annotations/instances_val2017.json"
        )
