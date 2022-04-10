import os
import argparse
from pathlib import Path
from prepare_fn import Mscoco14Prepare, Mscoco17Prepare, KittiPrepare, VirtualKittiPrepare, SynscapesPrepare, CityscapesPrepare

def main():

    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--datasets", nargs="+", help="<Required> Set flag", required=True)
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    args = parser.parse_args()

    print(args.datasets)
    for name in args.datasets:
        print(f"Process {name}")
        if name == "mscoco14":
            assert args.shift == "no"
            assert args.scale == "no"
            Mscoco14Prepare().prepare()
        elif name == "mscoco17":
            assert args.shift == "no"
            assert args.scale == "no"
            Mscoco17Prepare().prepare()
        elif name == "kitti":
            KittiPrepare(shift=args.shift, scale=args.scale).prepare()
        elif name == "virtual_kitti":
            VirtualKittiPrepare(shift=args.shift, scale=args.scale).prepare()
        elif name == "synscapes":
            SynscapesPrepare(shift=args.shift, scale=args.scale).prepare()
        elif name == "cityscapes":
            CityscapesPrepare(shift=args.shift, scale=args.scale).prepare()
        else:
            raise ValueError
            

if __name__ == "__main__":
    main()