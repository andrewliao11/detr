```python
python main.py --dataset_file viper --coco_path ~/datasets/viper/coco_format --batch_size 4
python main.py --dataset_file kitti --coco_path ~/datasets/kitti/coco_format --batch_size 4
python main.py --dataset_file virtual_kitti --coco_path ~/datasets/virtual_kitti/coco_format --batch_size 4

python main.py --dataset_file coco14 --coco_path ~/datasets/mscoco14/coco_format --batch_size 4
python main.py --dataset_file coco17 --coco_path ~/datasets/mscoco17/coco_format --batch_size 4

python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_file coco14 --coco_path ~/datasets/mscoco14/coco_format --batch_size 4 --lr_drop 100 --epochs 150 --output_dir /results --use_wandb
python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --dataset_file coco17 --coco_path ~/datasets/mscoco17/coco_format --batch_size 4 --lr_drop 100 --epochs 150 --output_dir /results --use_wandb
```


Sample results on mscoco2017 at [here](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f)
