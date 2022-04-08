# Steps to Run


## Preparation
This will prepare the specified dataset into desired COCO structure at `~/datasets/{$dataset_name}`
```bash
$ cd $detr_root/datasets_preparation
$ python {$dataset_name}_convert_to_coco.py
```

## Run Training 

```bash
$ cd $detr_root
$ python -m torch.distributed.launch --nproc_per_node=${n_gpus} --use_env main.py --dataset_file ${dataset_name} --coco_path ~/datasets/${dataset_name}/coco_format --batch_size 4 --lr_drop 100 --epochs 150 --output_dir /results
```

- [Suggested] You can add `--use_wandb` to log the metrics on wandb server.
- [Reference] Sample results on mscoco2017 at [here](https://gist.github.com/szagoruyko/b4c3b2c3627294fc369b899987385a3f)


### Max batch size for 16GB mem GPU
```yaml
mscoco14: 4
mscoco17: 4
kitti: 16
virtual_kitti: 16
```

## Conda env
```bash
$ conda create -n label-translation python=3.7.11
$ conda activate label-translation
$ pip install hydra-core hydra_colorlog --upgrade
$ pip install wandb
$ conda install pandas
$ conda install -c anaconda pillow 
$ pip install ipdb
$ conda install -c conda-forge pycocotools 
$ conda install -c anaconda scipy 
```
