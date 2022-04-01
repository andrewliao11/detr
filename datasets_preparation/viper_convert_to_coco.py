import os
import json
import pandas as pd

from PIL import Image
from pathlib import Path
import ipdb


split = "train"


dataset_root = Path("/datasets/viper")
target_dataset_root = os.environ['HOME'] / Path("datasets/viper/coco_format")


os.makedirs(target_dataset_root / f"{split}/data", exist_ok=True)


os.chdir(target_dataset_root / f"{split}/data")

 
image_paths = []
for p in dataset_root.glob(f"train/img/*/*.png"):

    if Path(str(p.with_suffix(".csv")).replace("img", "bb")).exists():
        tgt_name = p.name
        cmd = f"ln -s {p} {tgt_name}"

        if not Path(tgt_name).exists():
            os.system(cmd)
        image_paths.append(tgt_name)

os.chdir(target_dataset_root / split)


classes = pd.read_csv(dataset_root / "classes.csv")

labels = {
    "info": {},
    "licenses": [],
    "categories": [{
        "id": int(id),
        "name": c,
        "supercategory": "all"
    } for id, c in zip(classes["id"], classes["classname"])],
    "images": [],
    "annotations": []
}

image_root = target_dataset_root / f"{split}/data"
image_id = 1
file_name_to_image_id = {}
file_name_to_hw = {}
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
for anno_p in dataset_root.glob("train/bb/*/*.csv"):
    
    file_name = anno_p.with_suffix(".png").name
    if file_name in file_name_to_image_id:
        try:
            df = pd.read_csv(anno_p, sep=",", index_col=False, header=None)
            
            image_id = file_name_to_image_id[file_name]


            for i, row in df.iterrows():
                
                category_id = int(row[1])
                left, top, right, bottom = row[2:6]


                x_top_left, y_top_left, width, height = left, top, right-left, bottom-top

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
        except pd.errors.EmptyDataError:
            # empty data frame
            pass 


json.dump(labels, open(f"labels.json", "w"))
