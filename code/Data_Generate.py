#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import os
from einops import rearrange, repeat, reduce
from argument import Transform
from spectral import *
from spectral import open_image
import random
import math
from sklearn.metrics import normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

def cutting_img_mask(img, mask, cutting=224):
    while (1):
        xx = random.randint(0, img.shape[0] - cutting)
        yy = random.randint(0, img.shape[1] - cutting)
        patch_img = img[xx:xx + cutting, yy:yy + cutting]
        patch_mask = mask[xx:xx + cutting, yy:yy + cutting]
        if patch_mask.sum() != 0: break
    img = patch_img
    mask = patch_mask
    return img, mask

def crop_spectral(img, spectral_channel=32):
    orign_channel = img.shape[-1]
    if orign_channel <= spectral_channel:
        return img
    else:
        start_idx = np.random.choice(np.arange(orign_channel-32))
        img = img[:,:,start_idx:start_idx+spectral_channel]
    return img

class Data_Generate_Bile(Dataset):
    def __init__(self, img_paths, seg_paths, transform=None,
                 channels=None, principal_bands_num=-1,
                 cutting=-1, hyperhsi=False, cropspectral=-1):
        self.img_paths = img_paths
        self.seg_paths = seg_paths
        self.hyperhsi = hyperhsi
        self.transform = transform
        self.channels = channels
        self.principal_bands_num = principal_bands_num
        self.cutting = cutting
        self.cropspectral = cropspectral

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = envi.open(img_path)[:, :, :]

        mask_path = self.seg_paths[index]
        mask = (cv2.imread(mask_path, 0) / 255).astype(np.uint8)
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        if self.transform != None:
            img, mask = self.transform((img, mask))
        if self.cutting > 0:
            img, mask = cutting_img_mask(img, mask, self.cutting)

        mask = mask.astype(np.uint8)

        img = img[:, :, self.channels] if self.channels is not None else img
        if self.cropspectral > 0:
            img = crop_spectral(img, self.cropspectral)

        if self.hyperhsi:
            pc = principal_components(img)
            pc_0999 = pc.reduce(fraction=0.999)
            img_pc = pc_0999.transform(img)
            img_pca = img_pc[:, :, 0]
            Ws = dict()
            for i in range(img.shape[-1]):
                Ws[i] = (normalized_mutual_info_score(img_pca.reshape(-1), img[:, :, i].reshape(-1)))
            indexs = sorted(Ws.items(), key=lambda x: x[1], reverse=True)
            channels = [indexs[i][0] for i in range(16)]
            img = img[:, :, channels]

        if self.principal_bands_num < img.shape[-1] and self.principal_bands_num > 0:
            ######## caculate mutual_infor to choose principal component spectral bands image ##########
            pc = principal_components(img)
            pc_0999 = pc.reduce(fraction=0.999)
            img_pc = pc_0999.transform(img)
            img = img_pc[:, :, :self.principal_bands_num]
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.transpose(img, (2, 0, 1))

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        return img, mask[None]

    def __len__(self):
        return len(self.img_paths)

class Data_Generate_Gastric(Dataset):
    def __init__(self, img_paths, transform=None,
                 channels=None, principal_bands_num=-1,
                 cutting=-1, reshape=None, hyperhsi=False,
                 cropspectral=-1):

        self.img_paths = img_paths
        self.cropspectral = cropspectral
        self.transform = transform
        self.channels = channels
        self.principal_bands_num = principal_bands_num
        self.cutting = cutting
        self.reshape = reshape
        self.hyperhsi = hyperhsi

    def __getitem__(self, index):
        img_path = f"{self.img_paths[index]}.npy"
        img = np.load(img_path)[:, :, :]
        mask_path = f"{self.img_paths[index]}_mask.png"
        mask = (cv2.imread(mask_path, 0) / 255).astype(np.uint8)
        if self.reshape:
            img = cv2.resize(img, self.reshape)
            mask = cv2.resize(mask, self.reshape)
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        if self.transform != None:
            img, mask = self.transform((img, mask))
        if self.cutting > 0:
            img, mask = cutting_img_mask(img, mask, self.cutting)


        img = img[:, :, self.channels] if self.channels is not None else img
        if self.cropspectral > 0:
            img = crop_spectral(img, self.cropspectral)

        if self.hyperhsi:
            pc = principal_components(img)
            pc_0999 = pc.reduce(fraction=0.999)
            img_pc = pc_0999.transform(img)
            img_pca = img_pc[:, :, 0]
            Ws = dict()
            for i in range(img.shape[-1]):
                Ws[i] = (normalized_mutual_info_score(img_pca.reshape(-1), img[:, :, i].reshape(-1)))
            indexs = sorted(Ws.items(), key=lambda x: x[1], reverse=True)
            channels = [indexs[i][0] for i in range(16)]
            img = img[:, :, channels]

        if self.principal_bands_num < img.shape[-1] and self.principal_bands_num > 0:
            ######## caculate mutual_infor to choose principal component spectral bands image ##########
            pc = principal_components(img)
            pc_0999 = pc.reduce(fraction=0.999)
            img_pc = pc_0999.transform(img)
            img = img_pc[:, :, :self.principal_bands_num]
            img = np.transpose(img, (2, 0, 1))
        else:
            img = np.transpose(img, (2, 0, 1))

        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        return img, mask[None]

    def __len__(self):
        return len(self.img_paths)