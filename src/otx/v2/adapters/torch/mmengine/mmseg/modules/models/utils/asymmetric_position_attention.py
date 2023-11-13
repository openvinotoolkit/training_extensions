"""Asymmetric position attention module."""

# Copyright (c) 2019 MendelXu
# SPDX-License-Identifier: Apache-2.0

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional
from mmcv.cnn import ConvModule
from torch import nn

from .psp_layer import PSPModule


class AsymmetricPositionAttentionModule(nn.Module):
    """AsymmetricPositionAttentionModule.

    Reference: https://github.com/MendelXu/ANN.
    """

    def __init__(
        self,
        in_channels: int,
        key_channels: int,
        value_channels: int | None = None,
        psp_size: tuple[int, ...] | None = None,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
    ) -> None:
        """Asymmetric Position Attention Module.

        Args:
            in_channels (int): Number of input channels.
            key_channels (int): Number of channels for the query and key projections.
            value_channels (int, optional): Number of channels for the value projection.
                If not specified, defaults to `in_channels`.
            psp_size (tuple[int], optional): Pyramid pooling module sizes.
                If not specified, defaults to `(1, 3, 6, 8)`.
            conv_cfg (dict, optional): Dictionary to configure the convolutional layers.
                If not specified, defaults to `None`.
            norm_cfg (dict, optional): Dictionary to configure the normalization layers.
                If not specified, defaults to `{"type": "BN"}`.

        """
        super().__init__()

        if psp_size is None:
            psp_size = (1, 3, 6, 8)
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}

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
            act_cfg={"type": "ReLU"},
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
            act_cfg={"type": "ReLU"},
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        batch_size, _, _ = x.size(0), x.size(2), x.size(3)

        query_key = self.query_key(x)

        key = self.key_psp(query_key)
        value = self.value_psp(self.value(x)).permute(0, 2, 1)
        query = query_key.view(batch_size, self.key_channels, -1).permute(0, 2, 1)

        similarity_scores = torch.matmul(query, key)
        similarity_scores = (self.key_channels**-0.5) * similarity_scores
        similarity_scores = torch.nn.functional.softmax(similarity_scores, dim=-1)

        y = torch.matmul(similarity_scores, value)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.value_channels, *x.size()[2:])
        y = self.out_conv(y)

        return x + y
