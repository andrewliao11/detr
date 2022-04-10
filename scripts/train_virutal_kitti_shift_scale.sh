cd ../datasets_preparation/
python prepare.py --datasets virtual_kitti 
python prepare.py --datasets virtual_kitti --shift 0.2-up
cd ../
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py dataset=virtual_kitti add_dataset=virtual_kitti add_dataset.path=datasets/virtual_kitti_shift-up-0.2/coco_format batch_size=16 lr_drop=100 epochs=200 use_wandb=True 
