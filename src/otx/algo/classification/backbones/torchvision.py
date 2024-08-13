# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvison model's Backbone Class."""

from typing import Literal

import torch
from torch import nn
from torchvision.models import get_model, get_model_weights

TVModelType = Literal[
    "alexnet",
    "convnext_base",
    "convnext_large",
    "convnext_small",
    "convnext_tiny",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_l",
    "efficientnet_v2_m",
    "efficientnet_v2_s",
    "googlenet",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "regnet_x_16gf",
    "regnet_x_1_6gf",
    "regnet_x_32gf",
    "regnet_x_3_2gf",
    "regnet_x_400mf",
    "regnet_x_800mf",
    "regnet_x_8gf",
    "regnet_y_128gf",
    "regnet_y_16gf",
    "regnet_y_1_6gf",
    "regnet_y_32gf",
    "regnet_y_3_2gf",
    "regnet_y_400mf",
    "regnet_y_800mf",
    "regnet_y_8gf",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "resnext50_32x4d",
    "swin_b",
    "swin_s",
    "swin_t",
    "swin_v2_b",
    "swin_v2_s",
    "swin_v2_t",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "wide_resnet101_2",
    "wide_resnet50_2",
]


class TorchvisionBackbone(nn.Module):
    """TorchvisionBackbone is a class that represents a backbone model from the torchvision library."""

    def __init__(
        self,
        backbone: TVModelType,
        pretrained: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        tv_model_cfg = {"name": backbone}
        if pretrained:
            tv_model_cfg["weights"] = get_model_weights(backbone)
        net = get_model(**tv_model_cfg)
        self.features = net.features

        last_layer = list(net.children())[-1]
        classifier_len = len(list(last_layer.children()))
        self.in_features = (
            next(iter(last_layer.children())).in_features if classifier_len >= 1 else last_layer.in_features
        )

    def forward(self, *args) -> torch.Tensor:
        """Forward pass of the model."""
        return self.features(*args)
