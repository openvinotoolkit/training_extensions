# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""CSPNeXtPAFPN (CSPNeXt Path Aggregation Feature Pyramid Network)."""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from otx.algo.detection.layers.csp_layer import CSPLayer
from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.conv_module import ConvModule
from otx.algo.modules.depthwise_separable_conv_module import DepthwiseSeparableConvModule


class CSPNeXtPAFPN(BaseModule):
    """Path Aggregation Network with CSPNeXt blocks.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 3.
        use_depthwise (bool): Whether to use depthwise separable convolution in blocks. Defaults to False.
        expand_ratio (float): Ratio to adjust the number of channels of the hidden layer. Default: 0.5
        upsample_cfg (dict): Config dict for interpolate layer. Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer. Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer. Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        out_channels: int,
        num_csp_blocks: int = 3,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        upsample_cfg: dict | None = None,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        act_cfg: dict | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        if upsample_cfg is None:
            upsample_cfg = {"scale_factor": 2, "mode": "nearest"}

        if norm_cfg is None:
            norm_cfg = {"type": "BN", "momentum": 0.03, "eps": 0.001}

        if act_cfg is None:
            act_cfg = {"type": "Swish"}

        if init_cfg is None:
            init_cfg = {
                "type": "Kaiming",
                "layer": "Conv2d",
                "a": math.sqrt(5),
                "distribution": "uniform",
                "mode": "fan_in",
                "nonlinearity": "leaky_relu",
            }

        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * 2,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
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
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )
            self.bottom_up_blocks.append(
                CSPLayer(
                    in_channels[idx] * 2,
                    in_channels[idx + 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    use_cspnext_block=True,
                    expand_ratio=expand_ratio,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )

        self.out_convs = nn.ModuleList()
        for i in range(len(in_channels)):
            self.out_convs.append(
                conv(in_channels[i], out_channels, 3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            )

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        if len(inputs) != len(self.in_channels):
            msg = "The length of input features is not equal to the length of in_channels"
            raise ValueError(msg)

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
