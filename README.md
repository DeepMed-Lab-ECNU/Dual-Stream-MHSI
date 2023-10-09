# Factor Space and Spectrum for Medical Hyperspectral Image Segmentation (MICCAI 2023)
by Boxiang Yun, Qingli Li, Lubov Mitrofanova, Chunhua Zhou & Yan Wang*

## Introduction
Official code for "[Factor Space and Spectrum for Medical Hyperspectral Image Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_15)". (MICCAI 2023)

## Requirements
This repository is based on PyTorch 1.12.0, CUDA 11.３, and Python 3.９.７. All experiments in our paper were conducted on NVIDIA GeForce RTX 3090 GPU with an identical experimental setting.

## Usage
We provide `code`, `dataset`, and `model` for the MDC dataset.

The official dataset can be found at [MDC](http://bio-hsi.ecnu.edu.cn/). However, due to its size, we also provide preprocessed [data](https://www.kaggle.com/datasets/hfutybx/mhsi-choledoch-dataset-preprocessed-dataset) (including denoising and resize operations) for reproducing our paper experiments." 

Download the dataset and move to the dataset fold.

To train a model,
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1\
train_mdc.py  -r /dataset
-b 4 \
-spe_c 60 \
-b_group 15 \
-link_p 0 0 1 0 1 0 \
-sdr 4 4 \
-hw 320 256 \
-msd 4 4 \
-name Dual_MHSI \
```

To test a model,
```
CUDA_VISIBLE_DEVICES=0 python eval_seg_sst.py  -r /dataset \
-spe_c 60 \
-b_group 15 \
-link_p 0 0 1 0 1 0 \
-sdr 4 4 \
-hw 320 256 \
-msd 4 4 \
--pretrained_model ./bileseg-checkpoint/Dual_MHSI/best_epoch63_dice0.7547.pth
```
## Citation
If you find these projects useful, please consider citing:

```bibtex
@InProceedings{10.1007/978-3-031-43901-8_15,
author="Yun, Boxiang
and Li, Qingli
and Mitrofanova, Lubov
and Zhou, Chunhua
and Wang, Yan",
title="Factor Space and Spectrum for Medical Hyperspectral Image Segmentation",
booktitle="Medical Image Computing and Computer Assisted Intervention -- MICCAI 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="152--162",
}
```
## Acknowledgements
Some modules in our code were inspired by [Hamburger](https://github.com/Gsunshine/Enjoy-Hamburger) and [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch). We appreciate the effort of these authors to provide open-source code for the community. Hope our work can also contribute to related research.

## Questions
If you have any questions, welcome contact me at 'boxiangyun@gmail.com'
