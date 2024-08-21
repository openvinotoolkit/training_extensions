# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.necks.yolox_pafpn.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/necks/yolox_pafpn.py
"""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable

import torch
from torch import Tensor, nn

from otx.algo.detection.layers import CSPLayer
from otx.algo.modules.activation import Swish, build_activation_layer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import Conv2dModule, DepthwiseSeparableConvModule
from otx.algo.modules.norm import build_norm_layer


class YOLOXPAFPN(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``partial(nn.BatchNorm2d, momentum=0.03, eps=0.001)``.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``nn.Swish``.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        upsample_cfg: dict | None = None,
        normalization: Callable[..., nn.Module] = partial(nn.BatchNorm2d, momentum=0.03, eps=0.001),
        activation: Callable[..., nn.Module] = Swish,
        init_cfg: dict | list[dict] | None = None,
    ):
        upsample_cfg = upsample_cfg or {"scale_factor": 2, "mode": "nearest"}
        init_cfg = init_cfg or {
            "type": "Kaiming",
            "layer": "Conv2d",
            "a": math.sqrt(5),
            "distribution": "uniform",
            "mode": "fan_in",
            "nonlinearity": "leaky_relu",
        }

        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else Conv2dModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                Conv2dModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    normalization=build_norm_layer(normalization, num_features=in_channels[idx - 1]),
                    activation=build_activation_layer(activation),
                ),
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    normalization=normalization,
                    activation=activation,
                ),
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                conv(
                    in_channels[idx],
                    in_channels[idx],
                    3,
                    stride=2,
                    padding=1,
                    normalization=build_norm_layer(normalization, num_features=in_channels[idx]),
                    activation=build_activation_layer(activation),
                ),
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    normalization=normalization,
                    activation=activation,
                ),
            )

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                Conv2dModule(
                    in_channels[i],
                    out_channels,
                    1,
                    normalization=build_norm_layer(normalization, num_features=out_channels),
                    activation=build_activation_layer(activation),
                ),
            )

    def forward(self, inputs: tuple[Tensor]) -> tuple[Any, ...]:
        """Forward.

        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)  # noqa: S101

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](torch.cat([downsample_feat, feat_height], 1))
            outs.append(out)

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return tuple(outs)
