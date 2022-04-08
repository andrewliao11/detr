import os
import json
import pandas as pd

from PIL import Image
from pathlib import Path

from cityscape_labels import labels as LABELS
import ipdb


dataset_root = Path("/datasets/synscapes")
target_dataset_root = os.environ['HOME'] / Path("datasets/synscapes/coco_format")


os.makedirs(target_dataset_root / "data", exist_ok=True)



os.chdir(target_dataset_root / "data")

 
image_paths = []
for p in Path(dataset_root / "img/rgb").glob(f"*.png"):

    if p.name.startswith("."):
        continue

    tgt_name = p.name
    cmd = f"ln -s {p} {tgt_name}"
    if not Path(tgt_name).exists():
        os.system(cmd)
    image_paths.append(tgt_name)


os.chdir(target_dataset_root)


# https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

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

image_root = Path(target_dataset_root / "data")
image_id = 1
file_name_to_image_info= {}

for p in image_paths:

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
for anno_p in Path(dataset_root / "meta").glob("*.json"):
    
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
