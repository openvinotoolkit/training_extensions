# Copyright (C) 2019-2021 Xiangtai Lee
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class LocalAttentionModule(nn.Module):
    """Reference: https://github.com/lxtGH/GALD-DGCNet"""

    def __init__(self, num_channels, conv_cfg=None, norm_cfg=dict(type="BN")):
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
            act_cfg=dict(type="ReLU"),
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
            act_cfg=dict(type="ReLU"),
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
            act_cfg=dict(type="ReLU"),
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        y = self.dwconv1(x)
        y = self.dwconv2(y)
        y = self.dwconv3(y)
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=True)
        mask = self.sigmoid_spatial(y)

        out = x + x * mask

        return out
