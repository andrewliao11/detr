import os
import json
import pandas as pd

from PIL import Image
from pathlib import Path
import ipdb


KITTI_CLASSES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc", "DontCare"]


dataset_root = Path("/datasets/virtual_kitti")
target_dataset_root = os.environ['HOME'] / Path("datasets/virtual_kitti/coco_format")


os.makedirs(target_dataset_root / "data", exist_ok=True)



os.chdir(target_dataset_root / "data")


variants = ["clone", "morning", "sunset", "overcast", "fog", "rain"]

image_paths = []
for variant in variants:
    for p in dataset_root.glob(f"vkitti_1.3.1_rgb/*/{variant}/*.png"):

        tgt_name = "_".join(p.parts[-3:])
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
for anno_p in dataset_root.glob("vkitti_1.3.1_motgt/*.txt"):
    if any([v in str(anno_p) for v in variants]):

            world, variation = anno_p.stem.split("_")

            df = pd.read_csv(anno_p, sep=" ", index_col=False)

            for i, row in df.iterrows():
                anno_dict = row.to_dict()
                frame = anno_dict.pop("frame")

                file_name = f"{world}_{variation}_{str(frame).zfill(5)}.png"
                image_id = file_name_to_image_id[file_name]

                left, top, right, bottom = anno_dict["l"], anno_dict["t"], anno_dict["r"], anno_dict["b"]
                x_top_left, y_top_left, width, height = left, top, right-left, bottom-top
                category_id = KITTI_CLASSES.index(anno_dict["label"])


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

