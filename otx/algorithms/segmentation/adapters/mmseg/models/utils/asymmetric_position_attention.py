# Copyright (c) 2019 MendelXu
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from .psp_layer import PSPModule


class AsymmetricPositionAttentionModule(nn.Module):
    """Reference: https://github.com/MendelXu/ANN"""

    def __init__(
        self,
        in_channels,
        key_channels,
        value_channels=None,
        psp_size=(1, 3, 6, 8),
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels if value_channels is not None else in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.query_key = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.key_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type="ReLU"),
        )
        self.key_psp = PSPModule(psp_size, method="max")

        self.value = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.value_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type="ReLU"),
        )
        self.value_psp = PSPModule(psp_size, method="max")

        self.out_conv = ConvModule(
            in_channels=self.value_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None,
        )

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        query_key = self.query_key(x)

        key = self.key_psp(query_key)
        value = self.value_psp(self.value(x)).permute(0, 2, 1)
        query = query_key.view(batch_size, self.key_channels, -1).permute(0, 2, 1)

        similarity_scores = torch.matmul(query, key)
        similarity_scores = (self.key_channels**-0.5) * similarity_scores
        similarity_scores = F.softmax(similarity_scores, dim=-1)

        y = torch.matmul(similarity_scores, value)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.value_channels, *x.size()[2:])
        y = self.out_conv(y)

        out = x + y

        return out
