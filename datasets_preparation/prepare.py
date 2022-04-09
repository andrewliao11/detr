import os
import argparse
from pathlib import Path
from prepare_fn import Mscoco14Prepare, Mscoco17Prepare, KittiPrepare, VirtualKittiPrepare, SynscapesPrepare

def main():

    parser = argparse.ArgumentParser(description='Prepare datasets')
    parser.add_argument('--datasets', nargs='+', help='<Required> Set flag', required=True)
    args = parser.parse_args()

    print(args.datasets)
    for name in args.datasets:
        if name == "mscoco14":
            Mscoco14Prepare().prepare()
        elif name == "mscoco17":
            Mscoco17Prepare().prepare()
        elif name == "kitti":
            KittiPrepare().prepare()
        elif name == "virtual_kitti":
            VirtualKittiPrepare().prepare()
        elif name == "synscapes":
            SynscapesPrepare().prepare()
        else:
            raise ValueError
            

if __name__ == '__main__':
    main()