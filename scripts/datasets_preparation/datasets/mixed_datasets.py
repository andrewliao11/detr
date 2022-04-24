import os
import json

from pathlib import Path

from .base_prepare import BasePrepare


class MixedDatasetsPrepare(BasePrepare):
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
