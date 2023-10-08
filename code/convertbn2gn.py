#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023
@author: Boxiang Yun   School:ECNU   Email:boxiangyun@gmail.com
"""

import torch.nn as nn
from torchvision import models

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)

def convertbn2gn(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # Get current bn layer
            bn = get_layer(model, name)
            print(module)
            # Create new gn layer
            gn = nn.GroupNorm(8, bn.num_features) if bn.num_features % 8==0 else nn.GroupNorm(bn.num_features, bn.num_features)
            # Assign gn
            print("Swapping {} with {}".format(bn, gn))
            set_layer(model, name, gn)
    return model