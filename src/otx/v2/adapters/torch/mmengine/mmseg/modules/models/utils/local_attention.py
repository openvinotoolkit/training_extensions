"""Local attention module."""

# Copyright (C) 2019-2021 Xiangtai Lee
# SPDX-License-Identifier: MIT

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch.nn.functional
from mmcv.cnn import ConvModule
from torch import nn


class LocalAttentionModule(nn.Module):
    """LocalAttentionModule.

    Reference: https://github.com/lxtGH/GALD-DGCNet.
    """

    def __init__(self, num_channels: int, conv_cfg: dict | None = None, norm_cfg: dict | None = None) -> None:
        """Initializes LocalAttentionModule.

        Args:
        num_channels (int): The number of input channels.
        conv_cfg (dict | None): Optional configuration dictionary for the convolutional layers. Defaults to None.
        norm_cfg (dict | None): Optional configuration dictionary for the normalization layers. If None, batch
            normalization is used by default.

        """
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        super().__init__()

        self.num_channels = num_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.dwconv1 = ConvModule(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.num_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg={"type": "ReLU"},
        )
        self.dwconv2 = ConvModule(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.num_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg={"type": "ReLU"},
        )
        self.dwconv3 = ConvModule(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.num_channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg={"type": "ReLU"},
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        _, _, h, w = x.size()

        y = self.dwconv1(x)
        y = self.dwconv2(y)
        y = self.dwconv3(y)
        y = torch.nn.functional.interpolate(y, size=(h, w), mode="bilinear", align_corners=True)
        mask = self.sigmoid_spatial(y)

        return x + x * mask
