#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""
import numpy as np
import torch.nn as nn
import torch
from segmentation_models_pytorch import decoders
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder

from einops import rearrange, reduce
import torch.nn.functional as F
from hamburger.ham_spetral import get_hams

from lowrank.RTGB import DRTLM

class Hamburger(nn.Module):
    def __init__(self, in_c, args=None):
        super().__init__()
        ham_type = getattr(args, 'HAM_TYPE', 'NMF')#

        C = in_c
        self.norm = nn.BatchNorm1d(C)
        if ham_type == 'NMF':
            self.lower_bread = nn.Sequential(nn.Conv1d(C, C, 1),
                                             nn.ReLU(inplace=True))
        else:
            self.lower_bread = nn.Conv1d(C, C, 1)

        HAM = get_hams(ham_type)
        self.ham = HAM(args)
        self.ham.D = in_c

        self.upper_bread = nn.Conv1d(C, C, 1, bias=False)
        self.shortcut = nn.Sequential()

    def forward(self, x): #x:  # (b h w) c s
        # B, C, S, H, W = x.shape

        ham_x = self.norm(x)
        ham_x = self.lower_bread(ham_x)
        ham_x = self.ham(ham_x)

        out = F.relu(x + ham_x, inplace=True)
        return out

    def online_update(self, bases):
        if hasattr(self.ham, 'online_update'):
            self.ham.online_update(bases)

class SMD(nn.Module):
    def __init__(self, in_channels, hidden_feature, spe_reduction=1,
                 dim_reduction=4, kernel_size=1):
        super(SMD, self).__init__()
        self.dim_reduction = dim_reduction
        self.spe_reduction = spe_reduction

        self.depthwiseconv = nn.Conv2d(in_channels=in_channels, out_channels=hidden_feature, kernel_size=spe_reduction,
                                       stride=spe_reduction, groups=in_channels)
        self.ham = Hamburger(hidden_feature)
        self.spectral_ffn = nn.Sequential(
                                          nn.BatchNorm1d(hidden_feature),
                                          nn.Conv1d(hidden_feature, hidden_feature, kernel_size=kernel_size, stride=1,
                                                    padding=kernel_size // 2 if kernel_size != 1 else 0),
                                          nn.GELU(),
                                          nn.Conv1d(hidden_feature, hidden_feature, kernel_size=kernel_size, stride=1,
                                                    padding=kernel_size // 2 if kernel_size != 1 else 0),
                                          )


    def forward(self, x):
        b, c, s, h, w = x.shape
        x = rearrange(x, 'b c s h w -> (b s) c h w')# bs c h w
        x = self.depthwiseconv(x)#bs c h/4 w/4
        h_, w_ = x.shape[-2], x.shape[-1]

        x = rearrange(x, '(b s) c h w -> (b h w) c s', s=s)
        x = self.ham(x)
        o = self.spectral_ffn(x) + x
        o = rearrange(o, '(b h w) c s -> b c s h w', b=b, h=h_, w=w_)
        return o

class merge_block(nn.Module):
    def __init__(self, spa_in_channels, spe_in_channels, bands_group=4,
                 merge_spe_downsample=2):
        # concatenate spectral branch feature with spatial branch feature
        # "merge_spe_downsample" means downsample spectral dimension to adapt spatial feature map

        super(merge_block, self).__init__()
        self.bands_group = bands_group
        self.merge_spe_downsample = merge_spe_downsample

        self.merge_conv_spa = nn.Sequential(nn.Conv2d(spa_in_channels + spe_in_channels,
                                                      spa_in_channels, kernel_size=3, stride=1, padding=1))

    def forward(self, spa_input, spe_input):  # (b g) c h w , b c s h w

        spa_input = rearrange(spa_input, '(b g) c h w -> b g c h w', g=self.bands_group)
        spe2spa_input = reduce(spe_input, 'b c (s1 p) h w -> b s1 c h w', 'mean',
                               p=self.merge_spe_downsample)
        spa_input = self.merge_conv_spa(
            rearrange(torch.cat((spa_input, spe2spa_input), 2), 'b g c h w -> (b g) c h w'))

        return spa_input, spe_input

class LD(nn.Module):
    def __init__(self, in_channel, height=64, width=64, rank=4):
        super(LD, self).__init__()

        self.drtlm = DRTLM(rank, in_channel, height=height, width=width)

    def forward(self, spaspe_input, b):
        spa_output = self.drtlm(spaspe_input)
        spa_output = rearrange(spa_output, '(b g) c h w -> b g c h w', b=b).mean(1)
        return spa_output

class SpatialSpetralMixStream(nn.Module):
    def __init__(self, spectral_channels, linkpos=[0, 0, 1, 0, 1], bands_group=15,
                 backbone='resnet34', spectral_hidden_feature=32,
                 spatial_pretrain=False, spe_kernel_size=1,
                 attention_group='non', merge_spe_downsample=[2, 1],
                 spa_reduction=[4, 4], decode_choice='unet',
                 hw=[64, 64], rank=4):

        super(SpatialSpetralMixStream, self).__init__()
        self.spectral_channels = spectral_channels
        self.linkpos = linkpos
        self.spe_feature_dim = [1, spectral_hidden_feature * 1, spectral_hidden_feature * 2,
                                spectral_hidden_feature * 4,
                                spectral_hidden_feature * 8, spectral_hidden_feature * 16]

        self.bands_group = bands_group
        self.attention_group = attention_group

        spectrallayer_num = np.array(linkpos).nonzero()[0]
        ################### spatial branch encoder module ####################################
        self.spatial_backbone = get_encoder(name=backbone,
                                            in_channels= spectral_channels // bands_group,
                                            depth=5, weights='imagenet' if spatial_pretrain else None,
                                            output_stride=16 if decode_choice=='deeplabv3plus' else 32)
        self.spa_feature_dim = list(self.spatial_backbone.out_channels[1:])
        self.encoder_stages = self.spatial_backbone.get_stages()

        ################### fuse spatial and spectral feature map #############################
        self.merge_stages = nn.ModuleList([merge_block(self.spa_feature_dim[i - 1], self.spe_feature_dim[idx + 1],
                                                       bands_group=bands_group,
                                                       merge_spe_downsample=merge_spe_downsample[idx]) \
                                           for idx, i in enumerate(spectrallayer_num)])

        ################### spectral branch encoder module ####################################
        self.spe_encoder_stages = nn.ModuleList([SMD(
            self.spe_feature_dim[idx], self.spe_feature_dim[idx + 1],
            spa_reduction[idx],
            kernel_size=spe_kernel_size,
            ) for idx, i in enumerate(spectrallayer_num)])

        ################### low rank decomposition module ######################################
        if decode_choice != 'deeplabv3plus':
            self.spaspe_attention_head = LD(in_channel=self.spa_feature_dim[-1],
                                  height=hw[0] // (2 ** len(self.spa_feature_dim)),
                                  width=hw[1] // (2 ** len(self.spa_feature_dim)), rank=rank)
        else:
            self.spaspe_attention_head = LD(in_channel=self.spa_feature_dim[-1],
                                  height=hw[0] // (2 ** (len(self.spa_feature_dim)-1)),
                                  width=hw[1] // (2 ** (len(self.spa_feature_dim)-1)), rank=rank)

    def forward(self, input):
        features, spe_features = [], []
        spa_input = input
        spe_input = input.clone()
        spa_input = rearrange(spa_input, 'b (g c1) h w -> (b g) c1 h w', g=self.bands_group)

        x1, x2 = spa_input, spe_input[:,None]#b c s h w
        B = x2.shape[0]
        merge_position = 0

        for idx, encoder in enumerate(self.encoder_stages):
            x1 = encoder(x1)

            if self.attention_group == 'non' or idx == 0:
                ensemble_feature = reduce(x1, '(b g) c h w -> b c h w', 'mean', b=B)
                features.append(ensemble_feature)
            elif self.attention_group == 'lowrank':
                if idx == len(self.encoder_stages)-1:
                    ensemble_feature = self.spaspe_attention_head(x1, B)
                else:
                    ensemble_feature = reduce(x1, '(b g) c h w -> b c h w', 'mean', b=B)
                features.append(ensemble_feature)
            else:
                features.append(x1)
            ##### insert spectral feature map to spatial feature map
            if self.linkpos[idx]:
                x2 = self.spe_encoder_stages[merge_position](x2)
                x1, x2 = self.merge_stages[merge_position](x1, x2)
                merge_position = merge_position + 1
        return features

class SST_Seg_Dual(nn.Module):
    def __init__(self, spectral_channels, out_channels, linkpos=[0, 0, 1, 0, 1],
                 backbone='resnet34', spectral_hidden_feature=64, spatial_pretrain=False,
                 activation='sigmoid', decode_choice='unet', attention_group='non',
                 bands_group=1, spe_kernel_size=1, merge_spe_downsample=[2, 1],
                 spa_reduction=[4, 4], hw=[64,64], rank=4):
        """
        Args:
        spectral_channels: input's spectral_channels
        out_channels: number of segmentation classes
        linkpos (List): link spectral and spatial feature map with index position
        backbone: encoder backbone
        spectral_hidden_feature: spectral encoder's feature dimension
        spatial_pretrain: whether to use imagenet pretrain for spatial encoder
        activation: segmentation head activation
        decode_choice: eg: unet , deeplab....
        attention_group: whether to use low rank decomposition module
        bands_group: the number of division (group) of spectra
        spe_kernel_size: spectral encoder kernel size
        merge_spe_downsample: merge spectral and spatial feature for unifying map size
        spa_reduction (List): in spectral encoder, downsample ration of spatial dimension, the length of list must equal with linkpos nonzero number
        hw: the size of input image height and width
        rank: the hyperparameter for the low rank decomposition module
        """
        super(SST_Seg_Dual, self).__init__()
        decoder_channels = (256, 128, 64, 32, 16)
        assert spectral_channels % bands_group == 0
        spatial_channels = spectral_channels // bands_group
        self.backbone = backbone
        print(f"choose backbone is {backbone} decode_choice is {decode_choice}")
        self.encoder = SpatialSpetralMixStream(spectral_channels=spectral_channels,
                                                 spectral_hidden_feature=spectral_hidden_feature,
                                                 spatial_pretrain=spatial_pretrain,
                                                 backbone=backbone,
                                                 linkpos=linkpos,
                                                 bands_group=bands_group,
                                                 spe_kernel_size=spe_kernel_size,
                                                 spa_reduction=spa_reduction,
                                                 merge_spe_downsample=merge_spe_downsample,
                                                 decode_choice=decode_choice,
                                                 hw=hw,
                                                 rank=rank,
                                                 attention_group=attention_group)


        if decode_choice == 'unet':
            self.decoder = decoders.unet.decoder.UnetDecoder(
                encoder_channels=list([spatial_channels] + self.encoder.spa_feature_dim),
                decoder_channels=decoder_channels,
                n_blocks=5)
        elif decode_choice == 'fpn':
            self.decoder = decoders.fpn.decoder.FPNDecoder(
                encoder_channels=list([spatial_channels] + self.encoder.spa_feature_dim))
        elif decode_choice == 'unetplusplus':
            self.decoder = decoders.unetplusplus.decoder.UnetPlusPlusDecoder(
                encoder_channels=list([spatial_channels] + self.encoder.spa_feature_dim),
                decoder_channels = decoder_channels)
        elif decode_choice == 'deeplabv3plus':
            self.decoder = decoders.deeplabv3.decoder.DeepLabV3PlusDecoder(
                encoder_channels=list([spatial_channels] + self.encoder.spa_feature_dim),
                output_stride=16)

        if decode_choice=='unet' or decode_choice =='unetplusplus':
            upsampling = 1
        else:
            upsampling = 4

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1] if decode_choice=='unet' or decode_choice =='unetplusplus' else self.decoder.out_channels,
            out_channels=out_channels,
            activation=activation,
            kernel_size=3,
            upsampling=upsampling)

    def forward(self, input):
        features = self.encoder(input)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks
