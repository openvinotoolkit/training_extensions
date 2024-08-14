# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.backbones.cspnext.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/backbones/cspnext.py
"""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, ClassVar

from otx.algo.common.layers import SPPBottleneck
from otx.algo.detection.layers import CSPLayer
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from otx.algo.modules.norm import build_norm_layer
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm


class CSPNeXt(BaseModule):
    """CSPNeXt backbone used in RTMDet.

    Args:
        arch (str): Architecture of CSPNeXt, from {P5, P6}.
            Defaults to P5.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (2, 3, 4).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Defaults to -1.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        arch_ovewrite (list): Overwrite default arch settings.
            Defaults to None.
        spp_kernel_sizes: (tuple[int]): Sequential of kernel sizes of SPP
            layers. Defaults to (5, 9, 13).
        channel_attention (bool): Whether to add channel attention in each
            stage. Defaults to True.
        normalization (Callable[..., nn.Module]): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module] | None): Activation layer module.
            Defaults to ``nn.SiLU``.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        init_cfg (dict, list[dict]): Initialization config dict.
    """

    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp
    arch_settings: ClassVar = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 1024, 3, False, True],
        ],
        "P6": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 6, True, False],
            [512, 768, 3, True, False],
            [768, 1024, 3, False, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        out_indices: tuple[int, ...] = (2, 3, 4),
        frozen_stages: int = -1,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        arch_ovewrite: dict | None = None,
        spp_kernel_sizes: tuple[int, int, int] = (5, 9, 13),
        channel_attention: bool = True,
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] = nn.SiLU,
        norm_eval: bool = False,
        init_cfg: dict | None = None,
    ) -> None:
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
            arch_setting = arch_ovewrite  # type: ignore[assignment]

        if not set(out_indices).issubset(i for i in range(len(arch_setting) + 1)):
            msg = f"out_indices must be in range(0, len(arch_setting) + 1). But received {out_indices}"
            raise ValueError(msg)

        if frozen_stages not in range(-1, len(arch_setting) + 1):
            msg = f"frozen_stages must be in (-1, len(arch_setting) + 1). But received {frozen_stages}"
            raise ValueError(msg)

        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.use_depthwise = use_depthwise
        self.norm_eval = norm_eval
        conv = DepthwiseSeparableConvModule if use_depthwise else Conv2dModule
        self.stem = nn.Sequential(
            Conv2dModule(
                3,
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=2,
                normalization=build_norm_layer(
                    normalization,
                    num_features=int(arch_setting[0][0] * widen_factor // 2),
                ),
                activation=build_activation_layer(activation),
            ),
            Conv2dModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor // 2),
                3,
                padding=1,
                stride=1,
                normalization=build_norm_layer(
                    normalization,
                    num_features=int(arch_setting[0][0] * widen_factor // 2),
                ),
                activation=build_activation_layer(activation),
            ),
            Conv2dModule(
                int(arch_setting[0][0] * widen_factor // 2),
                int(arch_setting[0][0] * widen_factor),
                3,
                padding=1,
                stride=1,
                normalization=build_norm_layer(
                    normalization,
                    num_features=int(arch_setting[0][0] * widen_factor),
                ),
                activation=build_activation_layer(activation),
            ),
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
                    kernel_sizes=spp_kernel_sizes,
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
                use_cspnext_block=True,
                expand_ratio=expand_ratio,
                channel_attention=channel_attention,
                normalization=normalization,
                activation=activation,
            )
            stage.append(csp_layer)
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))
            self.layers.append(f"stage{i + 1}")

    def _freeze_stages(self) -> None:
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> None:
        """Set modules in training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Forward function."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
