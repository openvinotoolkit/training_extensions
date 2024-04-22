# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet backbones."""
from .resnet import ResNet
from .swin import SwinTransformer

__all__ = [
    "ResNet",
    "SwinTransformer",
]
