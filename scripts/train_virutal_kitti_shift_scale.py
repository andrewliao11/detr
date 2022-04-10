"""
python train_virutal_kitti_shift_scale.py --dataset virtual_kitti --add_dataset kitti --num_gpus4
"""

import os
import argparse
from pathlib import Path


def main():

    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--dataset", type=str, help="<Required> Set flag", required=True)
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    parser.add_argument("--add_dataset", type=str, help="<Required> Set flag", required=True)
    parser.add_argument("--num_gpus", type=int, default=4)

    args = parser.parse_args()

    os.chdir("../datasets_preparation")
    cmd = f"python prepare.py --datasets {args.dataset}"
    if args.shift != "no":
        cmd = f"{cmd} --shift {args.shift}"
    if args.scale != "no":
        cmd = f"{cmd} --scale {args.scale}"

    print(cmd)
    os.system(cmd)


    cmd = f"python prepare.py --datasets {args.add_dataset}"
    print(cmd)
    os.system(cmd)
    
    os.chdir("../")

    dataset_path = args.dataset
    if args.shift != "no":
        dataset_path = f"{dataset_path}_shift-{args.shift}"

    if args.scale != "no":
        dataset_path = f"{dataset_path}_scale-{args.scale}"

    dataset_path = f"datasets/{dataset_path}/coco_format"

    cmd = f"python -m torch.distributed.launch --nproc_per_node={args.num_gpus} --use_env main.py dataset={args.dataset} dataset.path={dataset_path} add_dataset={args.add_dataset} batch_size=16 lr_drop=100 epochs=200 use_wandb=True hydra.run.dir=/results"
    print(cmd)
    os.system(cmd)



if __name__ == '__main__':
    main()
