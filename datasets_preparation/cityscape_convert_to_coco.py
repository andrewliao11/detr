import os
import json
import pandas as pd

from PIL import Image
from pathlib import Path
import ipdb


KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]

dataset_root = Path("/datasets/kitti")
target_dataset_root = os.environ['HOME'] / Path("datasets/kitti/coco_format")


os.makedirs(target_dataset_root / "data", exist_ok=True)



os.chdir(target_dataset_root / "data")

 
image_paths = []
for p in Path(dataset_root / "Kitti/raw/training/image_2").glob(f"*.png"):

    tgt_name = p.name
    cmd = f"ln -s {p} {tgt_name}"
    if not Path(tgt_name).exists():
        os.system(cmd)
    image_paths.append(tgt_name)


os.chdir(target_dataset_root)


labels = {
    "info": {},
    "licenses": [],
    "categories": [{
        "id": i,
        "name": c,
        "supercategory": "all"
    } for i, c in enumerate(KITTI_CLASSES)],
    "images": [],
    "annotations": []
}

image_root = Path(target_dataset_root / "data")
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
for anno_p in Path(dataset_root / "Kitti/raw/training/label_2").glob("*.txt"):
    
    
    df = pd.read_csv(anno_p, sep=" ", index_col=False, header=None)
    file_name = anno_p.with_suffix(".png").name
    image_id = file_name_to_image_id[file_name]


    for i, row in df.iterrows():
        
        left, top, right, bottom = row[4:8]
        x_top_left, y_top_left, width, height = left, top, right-left, bottom-top
        category_id = KITTI_CLASSES.index(row[0])

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
