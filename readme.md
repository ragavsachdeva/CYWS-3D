# [The Change You Want to See (Now in 3D)](#)

[[Project Page]](#) [[arXiv]](https://arxiv.org/abs/2308.10417)

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2023

[Ragav Sachdeva](https://ragavsachdeva.github.io/), [Andrew Zisserman](https://scholar.google.com/citations?hl=en&user=UZ5wscMAAAAJ)

![results](demo_data/pred.gif)

## Installation

**Clone the repository**

```
git clone --recursive git@github.com:ragavsachdeva/CYWS-3D.git
```

**Install depedencies**

```
conda create -n cyws3d python=3.9 -y
conda activate cyws3d
conda install -c pytorch pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=11.3.1 -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d==0.7.1 -c pytorch3d --freeze-installed
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
pip install timm==0.6.12 jsonargparse matplotlib imageio loguru einops wandb easydict kornia==0.6.8 scipy etils mmdet==2.25.3
pip install segmentation-models-pytorch@git+https://github.com/ragavsachdeva/segmentation_models.pytorch.git@2cde92e776b0a074d5e2f4f6a50c68754f948015
```


## Datasets

KC-3D: `wget https://thor.robots.ox.ac.uk/cyws-3d/kc3d.tar`

RC-3D: `wget https://thor.robots.ox.ac.uk/cyws-3d/rc3d.tar`


## Pre-trained model

```
wget https://thor.robots.ox.ac.uk/cyws-3d/cyws-3d.ckpt.gz
gzip -d cyws-3d.ckpt.gz
```

## Example Usage

Please try running:

`python inference.py --load_weights_from ./cyws-3d.ckpt`

This should perform a batched inference on a set of example image pairs under various settings (see [this file](demo_data/input_metadata.yml)).

## Citation

```
@InProceedings{Sachdeva_ICCVW_2023,
    title = {The Change You Want to See (Now in 3D)},
    author = {Sachdeva, Ragav and Zisserman, Andrew},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year = {2023},
}
```
