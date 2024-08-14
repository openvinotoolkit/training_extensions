# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.backbones.resnext.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/backbones/resnext.py
"""

from __future__ import annotations

import math
from typing import ClassVar

from otx.algo.common.layers import ResLayer
from otx.algo.modules.norm import build_norm_layer
from torch import nn

from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class Bottleneck(_Bottleneck):
    """Bottleneck module for ResNeXt."""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        groups: int = 1,
        base_width: int = 4,
        base_channels: int = 64,
        **kwargs,
    ):
        """Bottleneck block for ResNeXt.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super().__init__(inplanes, planes, **kwargs)

        width = self.planes if groups == 1 else math.floor(self.planes * (base_width / base_channels)) * groups

        self.norm1_name, norm1 = build_norm_layer(self.normalization, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.normalization, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(self.normalization, self.planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False,
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=self.dilation,
            dilation=self.dilation,
            groups=groups,
            bias=False,
        )
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(width, self.planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

    def _del_block_plugins(self, plugin_names: list[str]) -> None:
        """Delete plugins for block if exist.

        Args:
            plugin_names (list[str]): List of plugins name to delete.
        """
        for plugin_name in plugin_names:
            del self._modules[plugin_name]


class ResNeXt(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    arch_settings: ClassVar[dict] = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self, groups: int = 1, base_width: int = 4, **kwargs):
        self.groups = groups
        self.base_width = base_width
        super().__init__(**kwargs)

    def make_res_layer(self, **kwargs) -> ResLayer:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(groups=self.groups, base_width=self.base_width, base_channels=self.base_channels, **kwargs)
