import os

from pathlib import Path
import ipdb


dataset_root = Path("/datasets/mscoco17")
target_dataset_root = os.environ['HOME'] / Path("datasets/mscoco17/coco_format/")



os.makedirs(target_dataset_root / "train", exist_ok=True)
os.chdir(target_dataset_root / "train")
cmd = f"ln -s {dataset_root / 'train2017'} data"
os.system(cmd)

cmd = f"ln -s {dataset_root / 'annotations/instances_train2017.json'} labels.json"
os.system(cmd)

os.makedirs(target_dataset_root / "val", exist_ok=True)
os.chdir(target_dataset_root / "val")
cmd = f"ln -s {dataset_root / 'val2017'} data"
os.system(cmd)

cmd = f"ln -s {dataset_root / 'annotations/instances_val2017.json'} labels.json"
os.system(cmd)
