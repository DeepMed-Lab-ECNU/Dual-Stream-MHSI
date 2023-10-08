#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""

import argparse
import pandas as pd
import json
import torch
from segmentation_models_pytorch import Unet, FPN, DeepLabV3Plus
import cv2

import numpy as np
from basemodel import SST_Seg_Dual
from convertbn2gn import convertbn2gn

import os
from torch.utils.data import DataLoader
from local_utils.seed_everything import seed_reproducer

from tqdm import tqdm
from Data_Generate import Data_Generate_Bile
from local_utils.metrics import iou, dice, sensitivity, specificity, hausdorff_distance_case, eval_f1score
import time

from monai.metrics import compute_hausdorff_distance
from monai.inferers import sliding_window_inference

def main(args):
    seed_reproducer(args.seed)

    root_path = args.root_path
    dataset_hyper = args.dataset_hyper
    dataset_mask = args.dataset_mask
    dataset_divide = args.dataset_divide
    batch = args.batch


    net_type = args.net
    principal_bands_num = args.principal_bands_num
    spectral_channels = args.spectral_channels

    spectral_hidden_feature = args.spectral_hidden_feature

    worker = args.worker
    decode_choice = args.decode_choice
    device = args.device
    classes = args.classes
    bands_group = args.bands_group
    link_position = args.link_position
    conver_bn2gn = args.conver_bn2gn
    backbone = args.backbone

    spe_kernel_size = args.spe_kernel_size
    attention_group = args.attention_group
    spa_reduction = args.spa_reduction

    cutting = args.cutting
    merge_spe_downsample = args.merge_spe_downsample
    hw = args.hw
    rank = args.rank

    images_root_path = os.path.join(root_path, dataset_hyper)
    mask_root_path = os.path.join(root_path, dataset_mask)
    dataset_json = os.path.join(root_path, dataset_divide)
    with open(dataset_json, 'r') as load_f:
        dataset_dict = json.load(load_f)
    test_files = dataset_dict['test']

    device = torch.device(device)

    val_transformer = None

    print(f'the number of testfiles is {len(test_files)}')

    test_images_path = [os.path.join(images_root_path, i) for i in test_files]
    test_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in test_files]
    print(test_images_path[:5])


    test_db = Data_Generate_Bile(test_images_path, test_masks_path, transform=val_transformer,
                                 principal_bands_num=principal_bands_num, cutting=cutting)
    test_loader = DataLoader(test_db, batch_size=batch, shuffle=False, num_workers=worker, drop_last=False)


    if net_type == 'backbone':
        if decode_choice == 'unet':
            model = Unet(in_channels=spectral_channels, encoder_name=backbone, encoder_weights=None,
                         classes=classes, activation='sigmoid').to(device)
        elif decode_choice == 'fpn':
            model = FPN(in_channels=spectral_channels, encoder_name=backbone, encoder_weights=None,
                         classes=classes, activation='sigmoid').to(device)
        elif decode_choice == 'deeplabv3plus':
            model = DeepLabV3Plus(in_channels=spectral_channels, encoder_name=backbone, encoder_weights=None,
                         classes=classes, activation='sigmoid').to(device)
    elif net_type == 'dual':
        model = SST_Seg_Dual(spectral_channels=spectral_channels,
                        out_channels=classes, spectral_hidden_feature=spectral_hidden_feature,
                        decode_choice=decode_choice,
                        backbone=backbone, bands_group=bands_group, linkpos=link_position,
                        spe_kernel_size=spe_kernel_size, spa_reduction=spa_reduction,
                        attention_group=attention_group, merge_spe_downsample=merge_spe_downsample,
                        hw=hw, rank=rank).to(device)
        if conver_bn2gn:
            model = convertbn2gn(model)
            model = model.to(device)
    else:
        raise ValueError("Oops! That was no valid model.Try again...")

    model.load_state_dict(torch.load(args.pretrained_model, map_location='cuda:0'), strict=True)

    history = {'test_iou':[], 'test_dice':[], 'test_hausf':[], 'test_iou_var':[], 'test_dice_var':[], 'test_hausf_var':[],
               'inference_time':[], 'params_M':[], 'MACs_G':[]}

    # with torch.cuda.device(0):
    #     macs, params = get_model_complexity_info(model, (60, 256, 320), as_strings=True, print_per_layer_stat=False, verbose=True)
    #     print(f"Computational complexity: , {macs}, {params}")

    history['params_M'] = 0.#params
    history['MACs_G'] = 0.#macs

    labels, outs = [], []
    imgs = []
    start_time = time.time()

    print('now start test ...')
    model.eval()

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(test_loader)):

            x1, label = sample
            x1, label = x1.to(device), label.to(device)
            out = model(x1)
            out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
            outs.extend(out)
            labels.extend(label)
            imgs.extend(x1.cpu().detach().numpy())

    end_time = time.time()
    inference_time = end_time - start_time

    outs, labels = np.array(outs), np.array(labels)
    outs = np.where(outs > 0.5, 1, 0)
    print(outs.shape, labels.shape)
    if args.vis:
        os.makedirs(f"{'/'.join(args.pretrained_model.split('/')[:-1])}/vis/", exist_ok=True)
        dices = np.array([dice(l, o) for l, o in zip(labels, outs)])
        _ = [cv2.imwrite(f"{'/'.join(args.pretrained_model.split('/')[:-1])}/vis/{test_files[i]}_{dices[i]}.jpg", (outs[i][0]*255).astype(np.uint8)) for i in range(len(outs))]
        _ = [cv2.imwrite(f"{'/'.join(args.pretrained_model.split('/')[:-1])}/vis/{test_files[i]}.jpg", (labels[i][0]*255).astype(np.uint8)) for i in range(len(labels))]


    test_iou = np.array([iou(l, o) for l, o in zip(labels, outs)]).mean()
    test_dice = np.array([dice(l, o) for l, o in zip(labels, outs)]).mean()
    test_iou_std = np.array([iou(l, o) for l, o in zip(labels, outs)]).std()
    test_dice_std = np.array([dice(l, o) for l, o in zip(labels, outs)]).std()

    test_hausfs = compute_hausdorff_distance(outs, labels)
    test_hausf_mean, test_hausf_std = test_hausfs.mean(), test_hausfs.std()

    print(f"dice is {test_dice} and iou is {test_iou}")

    history['test_iou'].append(test_iou)
    history['test_dice'].append(test_dice)
    history['test_hausf'].append(test_hausf_mean)
    history['test_iou_var'].append(test_iou_std)
    history['test_dice_var'].append(test_dice_std)
    history['test_hausf_var'].append(test_hausf_std)
    history['inference_time'].append(inference_time)

    history_pd = pd.DataFrame(history)
    history_pd.to_csv(os.path.join(f"{'/'.join(args.pretrained_model.split('/')[:-1])}", f'log_eval.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--root_path', '-r', type=str, default='./mdc_dataset/MDC')
    parser.add_argument('--dataset_hyper', '-dh', type=str, default='MHSI')
    parser.add_argument('--dataset_mask', '-dm', type=str, default='Mask')
    parser.add_argument('--dataset_divide', '-dd', type=str, default='train_val_test.json')
    parser.add_argument('--device', '-dev', type=str, default='cuda:0')

    parser.add_argument('--pretrained_model', '-pm', type=str,
                        default=None)
    parser.add_argument('--worker', '-nw', type=int,
                        default=4)

    parser.add_argument('--batch', '-b', type=int, default=1)
    parser.add_argument('--spectral_hidden_feature', '-shf', default=64, type=int)
    parser.add_argument('--rank', '-rank', type=int, default=4)
    parser.add_argument('--spectral_channels', '-spe_c', default=60, type=int)
    parser.add_argument('--principal_bands_num', '-pbn', default=-1, type=int)
    parser.add_argument('--conver_bn2gn', '-b2g', action='store_true', default=False)

    parser.add_argument('--decode_choice', '-de_c', default='unet', choices=['unet', 'fpn', 'deeplabv3plus'])

    parser.add_argument('--classes', '-c', type=int, default=1)
    parser.add_argument('--bands_group', '-b_group', type=int, default=1)
    parser.add_argument('--link_position', '-link_p', type=int, default=[0, 0, 0, 1, 0, 1], nargs='+')
    parser.add_argument('--spe_kernel_size', '-sks', type=int, default=1)#, nargs='+')
    parser.add_argument('--attention_group', '-att_g', type=str, default='lowrank', choices=['non', 'lowrank'])
    parser.add_argument('--vis', '-vis', action='store_true', default=False)
    parser.add_argument('--hw', '-hw', type=int, default=[128, 128], nargs='+')
    parser.add_argument('--spa_reduction', '-sdr', type=int, default=[4, 4], nargs='+')

    parser.add_argument('--cutting', '-cut', default=-1, type=int)
    parser.add_argument('--merge_spe_downsample', '-msd', type=int, default=[2, 1], nargs='+')
    parser.add_argument('--net', '-n', default='dual', type=str, choices=['backbone', 'dual'])
    parser.add_argument('--backbone', '-backbone', default='resnet34', type=str)

    args = parser.parse_args()

    main(args)