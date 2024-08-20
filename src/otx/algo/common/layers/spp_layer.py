# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.backbones.csp_darknet.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/backbones/csp_darknet.py
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch
from otx.algo.modules.activation import Swish, build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule
from otx.algo.modules.norm import build_norm_layer
from torch import Tensor, nn


class SPPBottleneck(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (tuple[int]): Sequential of kernel sizes of pooling
            layers. Default: (5, 9, 13).
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to ``Swish``.
        init_cfg (dict, list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (5, 9, 13),
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] | None = Swish,
        init_cfg: dict | list[dict] | None = None,
    ):
        super().__init__(init_cfg=init_cfg)
        mid_channels = in_channels // 2
        self.conv1 = Conv2dModule(
            in_channels,
            mid_channels,
            1,
            stride=1,
            normalization=build_norm_layer(normalization, num_features=mid_channels),
            activation=build_activation_layer(activation),
        )
        self.poolings = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels = mid_channels * (len(kernel_sizes) + 1)
        self.conv2 = Conv2dModule(
            conv2_channels,
            out_channels,
            1,
            normalization=build_norm_layer(normalization, num_features=out_channels),
            activation=build_activation_layer(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        x = self.conv1(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = torch.cat([x] + [pooling(x) for pooling in self.poolings], dim=1)
        return self.conv2(x)
