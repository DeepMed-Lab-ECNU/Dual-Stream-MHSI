#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
import argparse
import json
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from segmentation_models_pytorch import Unet, FPN, DeepLabV3Plus
from segmentation_models_pytorch.utils.losses import DiceLoss

import numpy as np
from basemodel import SST_Seg_Dual
from convertbn2gn import convertbn2gn
from torch import optim
from torch.utils.data import DataLoader
import os
from local_utils.tools import save_dict, save_pyfile
from local_utils.seed_everything import seed_reproducer
import pandas as pd

import signal
from tqdm import tqdm
from Data_Generate import Data_Generate_Bile
from argument import Transform
from local_utils.misc import AverageMeter
from local_utils.dice_bce_loss import Dice_BCE_Loss
from local_utils.metrics import iou, dice, sensitivity, specificity, hausdorff_distance_case, eval_f1score
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.scheduler import CosineLRScheduler

def reduce_tensor(tensor):
    rt = torch.tensor(tensor).cuda()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= torch.cuda.device_count()
    return rt.cpu().numpy()

def main(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    seed_reproducer(args.seed)

    root_path = args.root_path
    dataset_hyper = args.dataset_hyper
    dataset_mask = args.dataset_mask
    dataset_divide = args.dataset_divide
    batch = args.batch

    lr = args.lr
    wd = args.wd
    use_aug = args.use_aug
    experiment_name = args.experiment_name
    output_path = args.output
    epochs = args.epochs
    use_half = args.use_half

    scheduler_type = args.scheduler
    spatial_pretrain = args.spatial_pretrain
    net_type = args.net
    principal_bands_num = args.principal_bands_num
    spectral_channels = args.spectral_channels

    lf = args.loss_function
    spectral_hidden_feature = args.spectral_hidden_feature

    worker = args.worker
    decode_choice = args.decode_choice
    classes = args.classes
    bands_group = args.bands_group
    link_position = args.link_position
    conver_bn2gn = args.conver_bn2gn
    backbone = args.backbone

    spe_kernel_size = args.spe_kernel_size
    spa_reduction = args.spa_reduction
    cutting = args.cutting
    merge_spe_downsample = args.merge_spe_downsample
    hw = args.hw
    rank = args.rank
    attention_group = args.attention_group
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group(backend='nccl')

    images_root_path = os.path.join(root_path, dataset_hyper)
    mask_root_path = os.path.join(root_path, dataset_mask)
    dataset_json = os.path.join(root_path, dataset_divide)
    with open(dataset_json, 'r') as load_f:
        dataset_dict = json.load(load_f)

    train_files = dataset_dict['train']
    val_files = dataset_dict['val']
    test_files = dataset_dict['test']

    transform = Transform( Rotate_ratio=0.2, Flip_ratio=0.2) if use_aug else None
    val_transformer = None

    if local_rank==0:
        print(f'the number of trainfiles is {len(train_files)}')
        print(f'the number of valfiles is {len(val_files)}')
        print(f'the number of testfiles is {len(test_files)}')

    train_images_path = [os.path.join(images_root_path, i) for i in train_files]
    train_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in train_files]
    val_images_path = [os.path.join(images_root_path, i) for i in val_files]
    val_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in val_files]
    test_images_path = [os.path.join(images_root_path, i) for i in test_files]
    test_masks_path = [os.path.join(mask_root_path, f'{i[:-4]}.png') for i in test_files]

    train_db = Data_Generate_Bile(train_images_path, train_masks_path, transform=transform,
                            principal_bands_num=principal_bands_num, cutting=cutting)
    train_sampler = DistributedSampler(train_db)
    train_loader = DataLoader(train_db, sampler=train_sampler, batch_size=batch, num_workers=worker, drop_last=True)

    val_db = Data_Generate_Bile(val_images_path, val_masks_path, transform=val_transformer,
                                principal_bands_num=principal_bands_num, cutting=cutting)
    val_sampler = DistributedSampler(val_db)
    val_loader = DataLoader(val_db, sampler=val_sampler, batch_size=batch, shuffle=False, num_workers=worker, drop_last=False)

    test_db = Data_Generate_Bile(test_images_path, test_masks_path, transform=val_transformer,
                                 principal_bands_num=principal_bands_num, cutting=cutting)
    test_sampler = DistributedSampler(test_db)
    test_loader = DataLoader(test_db, sampler=test_sampler, batch_size=batch, shuffle=False, num_workers=worker, drop_last=False)

    if local_rank==0:
        if os.path.exists(f'{output_path}/{experiment_name}') == False:
            os.mkdir(f'{output_path}/{experiment_name}')
        save_dict(os.path.join(f'{output_path}/{experiment_name}', 'args.csv'), args.__dict__)

    if net_type == 'backbone':
        print(f"choose backbone is {backbone} and spatial_pretrain is {spatial_pretrain}")
        if decode_choice == 'unet':
            model = Unet(in_channels=spectral_channels, encoder_name=backbone, encoder_weights='imagenet' if spatial_pretrain else None,
                         classes=classes, activation='sigmoid').to(device)
        elif decode_choice == 'fpn':
            model = FPN(in_channels=spectral_channels, encoder_name=backbone, encoder_weights='imagenet' if spatial_pretrain else None,
                         classes=classes, activation='sigmoid').to(device)
        elif decode_choice == 'deeplabv3plus':
            model = DeepLabV3Plus(in_channels=spectral_channels, encoder_name=backbone, encoder_weights='imagenet' if spatial_pretrain else None,
                         classes=classes, activation='sigmoid').to(device)
    elif net_type == 'dual':
        model = SST_Seg_Dual(spectral_channels=spectral_channels,
                        out_channels=classes, spectral_hidden_feature=spectral_hidden_feature,
                        spatial_pretrain=spatial_pretrain, decode_choice=decode_choice,
                        backbone=backbone, bands_group=bands_group, linkpos=link_position,
                        spe_kernel_size=spe_kernel_size, spa_reduction=spa_reduction,
                        merge_spe_downsample=merge_spe_downsample, hw=hw, rank=rank,
                        attention_group=attention_group).to(device)
    else:
        raise ValueError("Oops! That was no valid model.Try again...")

    if conver_bn2gn:
        model = convertbn2gn(model)
        model = model.to(device)


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank, find_unused_parameters=True)


    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    if scheduler_type == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    elif scheduler_type == 'warmup':
        scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, warmup_t=9, warmup_prefix=True)
    else:
        raise ValueError("Oops! That was no valid scheduler_type.Try again...")

    if lf == 'dice':
        criterion = DiceLoss()
    elif lf == 'dicebce':
        criterion = Dice_BCE_Loss(bce_weight=0.5, dice_weight=0.5)
    else:
        raise ValueError("Oops! That was no valid lossfunction.Try again...")

    history = {'epoch': [], 'LR': [], 'train_loss': [], 'val_loss': [], 'val_iou': [],
               'val_dice': [], 'test_iou':[], 'test_dice':[], }

    ############## shut down train with save model and safely exit
    stop_training = False

    def sigint_handler(signal, frame):
        print("Ctrl+c caught, stopping the training and saving the model...")
        nonlocal stop_training
        stop_training = True
        history_pd = pd.DataFrame(history)
        history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', 'log.csv'), index=False)

    signal.signal(signal.SIGINT, sigint_handler)
    # install AverageMeter calss in order to caculate loss and score(iou)
    if use_half:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    best_val = 0.0
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)

        train_losses = AverageMeter()
        val_losses = AverageMeter()
        if local_rank==0:
            print('now start train ..')
            print('epoch {}/{}, LR:{}'.format(epoch + 1, epochs, optimizer.param_groups[0]['lr']))
        train_losses.reset()
        model.train()

        try:
            for idx, sample in enumerate(tqdm(train_loader)):
                if stop_training:
                    break
                x1, label = sample
                x1, label = x1.to(device), label.to(device)

                out = model(x1)
                loss = criterion(out, label)

                if use_half:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        optimizer.zero_grad()
                        scaled_loss.backward()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                optimizer.step()
                train_losses.update(loss.item())

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, please reduce batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return
            else:
                raise e

        print('now start validation ...')
        model.eval()
        labels, outs = [], []
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(val_loader)):
                if stop_training:
                    break
                x1, label = sample
                x1, label = x1.to(device), label.to(device)

                out = model(x1)
                loss = criterion(out, label)

                val_losses.update(loss.item())
                out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
                outs.extend(out)
                labels.extend(label)
        outs, labels = np.array(outs), np.array(labels)
        outs = np.where(outs > 0.5, 1, 0)
        val_iou = np.array([iou(l, o) for l, o in zip(labels, outs)]).mean()
        val_dice = np.array([dice(l, o) for l, o in zip(labels, outs)]).mean()

        print('now start test ...')
        model.eval()

        labels, outs = [], []
        with torch.no_grad():
            for idx, sample in enumerate(tqdm(test_loader)):
                if stop_training:
                    break
                x1, label = sample
                x1, label = x1.to(device), label.to(device)

                out = model(x1)
                out, label = out.cpu().detach().numpy(), label.cpu().detach().numpy()
                outs.extend(out)
                labels.extend(label)

        outs, labels = np.array(outs), np.array(labels)
        outs = np.where(outs > 0.5, 1, 0)
        test_iou = np.array([iou(l, o) for l, o in zip(labels, outs)]).mean()
        test_dice = np.array([dice(l, o) for l, o in zip(labels, outs)]).mean()

        print('epoch {}/{}\t LR:{}\t train loss:{}\t val_dice:{}' \
              .format(epoch + 1, epochs, optimizer.param_groups[0]['lr'], train_losses.avg, val_dice))

        history['train_loss'].append(reduce_tensor(train_losses.avg))
        history['val_loss'].append(reduce_tensor(val_losses.avg))

        history['val_iou'].append(reduce_tensor(val_iou))
        history['val_dice'].append(reduce_tensor(val_dice))
        history['test_iou'].append(reduce_tensor(test_iou))
        history['test_dice'].append(reduce_tensor(test_dice))

        history['epoch'].append(epoch + 1)
        history['LR'].append(optimizer.param_groups[0]['lr'])

        if scheduler_type == 'warmup':
            scheduler.step(epoch)
        else:
            scheduler.step()

        if stop_training:
            torch.save(model.state_dict(),
                       os.path.join(f'{output_path}/{experiment_name}', 'final_{}.pth'.format(val_losses.avg)))
            break

        if best_val <= reduce_tensor(val_dice):
            if local_rank==0:
                if epoch > 0 and os.path.exists(save_path):
                    os.remove(save_path)

            best_val = reduce_tensor(val_dice)
            torch.save(model.module.state_dict(),
                       os.path.join(f'{args.output}/{args.experiment_name}',
                                    f'best_epoch{epoch}_dice{best_val:.4f}.pth'))
            save_path = os.path.join(f'{args.output}/{args.experiment_name}',
                                     f'best_epoch{epoch}_dice{best_val:.4f}.pth')

        if local_rank == 0:
            history_pd = pd.DataFrame(history)
            history_pd.to_csv(os.path.join(f'{output_path}/{experiment_name}', f'log.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--root_path', '-r', type=str, default='./mdc_dataset/MDC')
    parser.add_argument('--dataset_hyper', '-dh', type=str, default='MHSI')
    parser.add_argument('--dataset_mask', '-dm', type=str, default='Mask')
    parser.add_argument('--dataset_divide', '-dd', type=str, default='train_val_test.json')

    parser.add_argument('--worker', '-nw', type=int,
                        default=4)
    parser.add_argument('--use_half', '-uh', action='store_true', default=False)
    parser.add_argument('--batch', '-b', type=int, default=8)

    parser.add_argument('--spatial_pretrain', '-sp', action='store_true', default=False)
    parser.add_argument('--lr', '-l', default=0.0003, type=float)
    parser.add_argument('--wd', '-w', default=5e-4, type=float)
    parser.add_argument('--spectral_hidden_feature', '-shf', default=64, type=int)

    parser.add_argument('--rank', '-rank', type=int, default=4)
    parser.add_argument('--spectral_channels', '-spe_c', default=60, type=int)
    parser.add_argument('--principal_bands_num', '-pbn', default=-1, type=int)
    parser.add_argument('--conver_bn2gn', '-b2g', action='store_true', default=False)
    parser.add_argument('--use_aug', '-aug', action='store_true', default=True)
    parser.add_argument('--output', '-o', type=str, default='./bileseg-checkpoint')
    parser.add_argument('--experiment_name', '-name', type=str, default='Dual_MHSI')
    parser.add_argument('--decode_choice', '-de_c', default='unet', choices=['unet', 'fpn', 'deeplabv3plus'])
    parser.add_argument('--epochs', '-e', type=int, default=100)
    parser.add_argument('--classes', '-c', type=int, default=1)
    parser.add_argument('--bands_group', '-b_group', type=int, default=1)
    parser.add_argument('--link_position', '-link_p', type=int, default=[0, 0, 0, 1, 0, 1], nargs='+')
    parser.add_argument('--loss_function', '-lf', default='dicebce', choices=['dicebce', 'bce'])
    parser.add_argument('--spe_kernel_size', '-sks', type=int, default=1)#, nargs='+')

    parser.add_argument('--hw', '-hw', type=int, default=[128, 128], nargs='+')
    parser.add_argument('--spa_reduction', '-sdr', type=int, default=[4, 4], nargs='+')
    parser.add_argument('--cutting', '-cut', default=-1, type=int)
    parser.add_argument('--merge_spe_downsample', '-msd', type=int, default=[2, 1], nargs='+')
    parser.add_argument('--scheduler', '-sc', default='cos', choices=['cos', 'warmup'])
    parser.add_argument('--net', '-n', default='dual', type=str, choices=['backbone', 'dual'])
    parser.add_argument('--backbone', '-backbone', default='resnet34', type=str)
    args = parser.parse_args()

    main(args)