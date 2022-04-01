import os

from pathlib import Path
import ipdb


dataset_root = Path("/datasets/mscoco14")
target_dataset_root = os.environ['HOME'] / Path("datasets/mscoco14/coco_format/")



os.makedirs(target_dataset_root / "train", exist_ok=True)
os.chdir(target_dataset_root / "train")
cmd = f"ln -s {dataset_root / 'train2014/train2014'} data"
os.system(cmd)

cmd = f"ln -s {dataset_root / 'instances_train-val2014/annotations/instances_train2014.json'} labels.json"
os.system(cmd)


os.makedirs(target_dataset_root / "val", exist_ok=True)
os.chdir(target_dataset_root / "val")
cmd = f"ln -s {dataset_root / 'val2014/val2014'} data"
os.system(cmd)

cmd = f"ln -s {dataset_root / 'instances_train-val2014/annotations/instances_val2014.json'} labels.json"
os.system(cmd)
