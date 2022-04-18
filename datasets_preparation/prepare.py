import os
import time
import argparse
from pathlib import Path
from prepare_fn import Mscoco14Prepare, Mscoco17Prepare, KittiPrepare, VirtualKittiPrepare, SynscapesPrepare, CityscapesPrepare, MixedDatasetsPrepare

def main():

    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--datasets", nargs="+", help="<Required> Set flag", required=True)
    parser.add_argument("--shift", type=str, default="no", help="[ratio(0.~1.)]-[direction(left,top,right,bottom)]")
    parser.add_argument("--scale", type=str, default="no", help="[ratio(0.~1.)]-[direction(up,down)]")
    args = parser.parse_args()

    print(args.datasets)
    t1 = time.time()

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
        elif name == "mixed_kitti_virtual_kitti":
            kitti_prepare = KittiPrepare(shift="no", scale="no")
            virtual_kitti_prepare = VirtualKittiPrepare(shift=args.shift, scale=args.scale)

            MixedDatasetsPrepare(kitti_prepare, virtual_kitti_prepare).prepare()
        elif name == "10to1_mixed_kitti_virtual_kitti":
            kitti_prepare = KittiPrepare(train_ratio=0.1, shift="no", scale="no")
            virtual_kitti_prepare = VirtualKittiPrepare(shift=args.shift, scale=args.scale)

            MixedDatasetsPrepare(kitti_prepare, virtual_kitti_prepare).prepare()
        elif name == "mixed_cityscapes_synscapes":
            cityscapes_prepare = CityscapesPrepare(shift="no", scale="no")
            synscapes_prepare = SynscapesPrepare(shift=args.shift, scale=args.scale)
            
            MixedDatasetsPrepare(cityscapes_prepare, synscapes_prepare).prepare()
        else:
            raise ValueError

    print(f"Takes {time.time() - t1} secs")
    

if __name__ == "__main__":
    main()
    