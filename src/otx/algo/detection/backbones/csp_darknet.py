# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.backbones.csp_darknet.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/backbones/csp_darknet.py
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable, ClassVar, Sequence

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from otx.algo.common.layers import SPPBottleneck
from otx.algo.detection.layers import CSPLayer
from otx.algo.modules.activation import Swish, build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from otx.algo.modules.norm import build_norm_layer


class Focus(nn.Module):
    """Focus width and height information into channel space.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_size (int): The kernel size of the convolution. Default: 1
        stride (int): The stride of the convolution. Default: 1
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to ``Swish``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] | None = Swish,
    ):
        super().__init__()
        self.conv = Conv2dModule(
            in_channels * 4,
            out_channels,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2,
            normalization=build_norm_layer(normalization, num_features=out_channels),
            activation=build_activation_layer(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

    def export(self, x: Tensor) -> Tensor:
        """Forward for export."""
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1, 2, w)
        x = x.reshape(b, c, x.shape[2], 2, -1, 2)
        half_h = x.shape[2]
        half_w = x.shape[4]
        x = x.permute(0, 5, 3, 1, 2, 4)
        x = x.reshape(b, c * 4, half_h, half_w)

        return self.conv(x)


class CSPDarknet(BaseModule):
    """CSP-Darknet backbone used in YOLOv5 and YOLOX.

    Args:
        arch (str): Architecture of CSP-Darknet, from {P5, P6}.
            Default: P5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Default: 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Default: (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False.
        arch_ovewrite(list): Overwrite default arch settings. Default: None.
        spp_kernal_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Default: (5, 9, 13).
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to ``Swish``.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict, list[dict], optional): Initialization config dict.
            Default: None.
    """

    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings: ClassVar = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 9, True, False],
            [256, 512, 9, True, False],
            [512, 1024, 3, False, True],
        ],
        "P6": [
            [64, 128, 3, True, False],
            [128, 256, 9, True, False],
            [256, 512, 9, True, False],
            [512, 768, 3, True, False],
            [768, 1024, 3, False, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        arch_ovewrite: list | None = None,
        spp_kernal_sizes: tuple[int, ...] = (5, 9, 13),
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] = Swish,
        norm_eval: bool = False,
        init_cfg: dict | list[dict] | None = None,
    ):
        init_cfg = init_cfg or {
            "type": "Kaiming",
            "layer": "Conv2d",
            "a": math.sqrt(5),
            "distribution": "uniform",
            "mode": "fan_in",
            "nonlinearity": "leaky_relu",
        }
        super().__init__(init_cfg=init_cfg)

        arch_setting = self.arch_settings[arch]
        if arch_ovewrite:
            arch_setting = arch_ovewrite
        assert set(out_indices).issubset(i for i in range(len(arch_setting) + 1))  # noqa: S101
        if frozen_stages not in range(-1, len(arch_setting) + 1):
            msg = f"frozen_stages must be in range(-1, len(arch_setting) + 1). But received {frozen_stages}"
            raise ValueError(msg)

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else Conv2dModule

        self.stem = Focus(
            3,
            int(arch_setting[0][0] * widen_factor),
            kernel_size=3,
            normalization=normalization,
            activation=activation,
        )
        self.layers = ["stem"]

        for i, (in_channels, out_channels, num_blocks, add_identity, use_spp) in enumerate(arch_setting):
            in_channels = int(in_channels * widen_factor)  # noqa: PLW2901
            out_channels = int(out_channels * widen_factor)  # noqa: PLW2901
            num_blocks = max(round(num_blocks * deepen_factor), 1)  # noqa: PLW2901
            stage = []
            conv_layer = conv(
                in_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                normalization=build_norm_layer(normalization, num_features=out_channels),
                activation=build_activation_layer(activation),
            )
            stage.append(conv_layer)
            if use_spp:
                spp = SPPBottleneck(
                    out_channels,
                    out_channels,
                    kernel_sizes=spp_kernal_sizes,
                    normalization=normalization,
                    activation=activation,
                )
                stage.append(spp)
            csp_layer = CSPLayer(
                out_channels,
                out_channels,
                num_blocks=num_blocks,
                add_identity=add_identity,
                use_depthwise=use_depthwise,
                normalization=normalization,
                activation=activation,
            )
            stage.append(csp_layer)
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def _freeze_stages(self) -> None:
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        """Make the model trainable."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tensor) -> tuple[Any, ...]:
        """Forward."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
