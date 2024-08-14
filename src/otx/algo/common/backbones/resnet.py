# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet ResNet."""

from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, ClassVar

import torch
import torch.utils.checkpoint as cp
from otx.algo.common.layers import ResLayer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.norm import build_norm_layer
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


class Bottleneck(BaseModule):
    """Bottleneck block for ResNet.

    If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
    it is "caffe", the stride-two layer is the first 1x1 conv layer
    """

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        normalization: Callable[..., nn.Module],
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
        with_cp: bool = False,
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.normalization = normalization

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.norm1_name, norm1 = build_norm_layer(normalization, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalization, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(normalization, planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=self.conv1_stride, bias=False)
        self.add_module(self.norm1_name, norm1)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )

        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: normalization layer after the first convolution layer."""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self) -> nn.Module:
        """nn.Module: normalization layer after the second convolution layer."""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self) -> nn.Module:
        """nn.Module: normalization layer after the third convolution layer."""
        return getattr(self, self.norm3_name)

    def forward(self, x: torch.Tensor) -> nn.Module:
        """Forward function."""

        def _inner_forward(x: torch.Tensor) -> nn.Module:
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = cp.checkpoint(_inner_forward, x) if self.with_cp and x.requires_grad else _inner_forward(x)

        return self.relu(out)


class ResNet(BaseModule):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``partial(build_norm_layer, nn.BatchNorm2d, requires_grad=True)``.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    arch_settings: ClassVar = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth: int,
        in_channels: int = 3,
        stem_channels: int | None = None,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: tuple[int, int, int, int] = (1, 2, 2, 2),
        dilations: tuple[int, int, int, int] = (1, 1, 1, 1),
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
        avg_down: bool = False,
        frozen_stages: int = -1,
        normalization: Callable[..., nn.Module] = partial(
            build_norm_layer,
            nn.BatchNorm2d,
            requires_grad=True,
        ),
        norm_eval: bool = True,
        with_cp: bool = False,
        zero_init_residual: bool = True,
        pretrained: str | bool | None = None,
        init_cfg: list[dict] | dict | None = None,
    ):
        super().__init__(init_cfg)
        self.zero_init_residual = zero_init_residual
        if depth not in self.arch_settings:
            msg = f"invalid depth {depth} for resnet"
            raise KeyError(msg)

        block_init_cfg = None
        self.init_cfg: list[dict] | dict | None = None
        if init_cfg and pretrained:
            msg = "init_cfg and pretrained cannot be specified at the same time"
            raise ValueError(msg)
        if isinstance(pretrained, str):
            warnings.warn("DeprecationWarning: pretrained is deprecated, please use init_cfg instead", stacklevel=2)
            self.init_cfg = {"type": "Pretrained", "checkpoint": pretrained}
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    {"type": "Kaiming", "layer": "Conv2d"},
                    {"type": "Constant", "val": 1, "layer": ["BatchNorm", "GroupNorm"]},
                ]
                if self.zero_init_residual:
                    block_init_cfg = {"type": "Constant", "val": 0, "override": {"name": "norm3"}}
        else:
            msg = "pretrained must be a str or None"
            raise TypeError(msg)

        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        if num_stages > 4 or num_stages < 1:
            msg = "num_stages must be in [1, 4]"
            raise ValueError(msg)
        self.strides = strides
        self.dilations = dilations
        if len(strides) != len(dilations) != num_stages:
            msg = "The length of strides, dilations and out_indices should be the same as num_stages"
            raise ValueError(msg)
        self.out_indices = out_indices
        if max(out_indices) >= num_stages:
            msg = "max(out_indices) should be smaller than num_stages"
            raise ValueError(msg)
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.normalization = normalization
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                avg_down=self.avg_down,
                with_cp=with_cp,
                normalization=normalization,
                init_cfg=block_init_cfg,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

    def make_res_layer(self, **kwargs) -> ResLayer:
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self) -> nn.Module:
        """nn.Module: the normalization layer named "norm1"."""
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels: int, stem_channels: int) -> None:
        self.conv1 = nn.Conv2d(
            in_channels,
            stem_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.norm1_name, norm1 = build_norm_layer(
            self.normalization,
            stem_channels,
            postfix=1,
        )
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            self.norm1.eval()
            for m in [self.conv1, self.norm1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward function."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode: bool = True) -> None:
        """Convert the model into training mode while keep normalization layer freezed."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
