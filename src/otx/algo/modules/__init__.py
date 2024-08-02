# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This module implementation is a code implementation copied or replaced from mmcv.cnn.bricks."""

from .activation import build_activation_layer
from .conv_module import Conv2dModule, Conv3dModule
from .depthwise_separable_conv_module import DepthwiseSeparableConvModule
from .norm import FrozenBatchNorm2d, build_norm_layer
from .padding import build_padding_layer

__all__ = [
    "build_activation_layer",
    "build_padding_layer",
    "build_norm_layer",
    "Conv2dModule",
    "Conv3dModule",
    "DepthwiseSeparableConvModule",
    "FrozenBatchNorm2d",
]
