import os
import argparse
from pathlib import Path
from prepare_fn import Mscoco14Prepare, Mscoco17Prepare, KittiPrepare, VirtualKittiPrepare, SynscapesPrepare, CityscapesPrepare

def main():

    parser = argparse.ArgumentParser(description="Run training")
    parser.add_argument("--dataset", type=str, help="<Required> Set flag", required=True)
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    parser.add_argument("--num_gpus", type=int, default=4)

    args = parser.parse_args()

    os.chdir("../datasets_preparation")
    cmd = f"python prepare.py --datasets {args.dataset}"
    os.system(cmd)
    if shift != "no":
        cmd = f"{cmd} --shift {args.shift}"
    if scale != "no":
        cmd = f"{cmd} --scale {args.scale}"
    os.system(cmd)
    
    os.chdir("../")

    add_dataset_path = args.dataset
    if shift != "no":
        add_dataset_path = f"{add_dataset_path}_shift-{args.shift}"

    if scale != "no":
        add_dataset_path = f"{add_dataset_path}_scale-{args.scale}"

    add_dataset_path = f"datasets/{add_dataset_path}/coco_format"

    cmd = f"python -m torch.distributed.launch --nproc_per_node={args.num_gpus} --use_env main.py dataset={args.dataset} add_dataset={args.dataset} add_dataset.path={add_dataset_path} --batch_size 16 --lr_drop 100 --epochs 200 "
    
    os.system(cmd)



if __name__ == '__main__':
    main()
