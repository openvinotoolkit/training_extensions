# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks/modules for semantic segmentation."""

from __future__ import annotations

from typing import Callable, ClassVar

import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d

from otx.algo.modules import Conv2dModule
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.norm import build_norm_layer


class PSPModule(nn.Module):
    """PSP module.

    Reference: https://github.com/MendelXu/ANN.
    """

    methods: ClassVar[dict[str, AdaptiveMaxPool2d | AdaptiveAvgPool2d]] = {
        "max": AdaptiveMaxPool2d,
        "avg": AdaptiveAvgPool2d,
    }

    def __init__(self, sizes: tuple = (1, 3, 6, 8), method: str = "max"):
        super().__init__()

        pool_block = self.methods[method]

        self.stages = nn.ModuleList([pool_block(output_size=(size, size)) for size in sizes])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """Forward."""
        batch_size, c, _, _ = feats.size()

        priors = [stage(feats).view(batch_size, c, -1) for stage in self.stages]

        return torch.cat(priors, -1)


class AsymmetricPositionAttentionModule(nn.Module):
    """AsymmetricPositionAttentionModule.

    Reference: https://github.com/MendelXu/ANN.
    """

    def __init__(
        self,
        in_channels: int,
        key_channels: int,
        value_channels: int | None = None,
        psp_size: tuple | None = None,
        normalization: Callable[..., nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels if value_channels is not None else in_channels
        if psp_size is None:
            psp_size = (1, 3, 6, 8)
        self.normalization = normalization
        self.query_key = Conv2dModule(
            in_channels=self.in_channels,
            out_channels=self.key_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            normalization=build_norm_layer(self.normalization, num_features=self.key_channels),
            activation=build_activation_layer(nn.ReLU),
        )
        self.key_psp = PSPModule(psp_size, method="max")

        self.value = Conv2dModule(
            in_channels=self.in_channels,
            out_channels=self.value_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            normalization=build_norm_layer(self.normalization, num_features=self.value_channels),
            activation=build_activation_layer(nn.ReLU),
        )
        self.value_psp = PSPModule(psp_size, method="max")

        self.out_conv = Conv2dModule(
            in_channels=self.value_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            normalization=build_norm_layer(self.normalization, num_features=self.in_channels),
            activation=None,
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
        similarity_scores = f.softmax(similarity_scores, dim=-1)

        y = torch.matmul(similarity_scores, value)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.value_channels, *x.size()[2:])
        y = self.out_conv(y)

        return x + y


class OnnxLpNormalization(torch.autograd.Function):
    """OnnxLpNormalization."""

    @staticmethod
    def forward(
        ctx: Callable,
        x: torch.Tensor,
        axis: int = 0,
        p: int = 2,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Forward."""
        del ctx, p  # These args are not used.
        denom = x.norm(2, axis, True).clamp_min(eps).expand_as(x)
        return x / denom

    @staticmethod
    def symbolic(
        g: torch.Graph,
        x: torch.Tensor,
        axis: int = 0,
        p: int = 2,
        eps: float = 1e-12,
    ) -> torch.Graph:
        """Symbolic onnxLpNormalization."""
        del eps  # These args are not used.
        return g.op("LpNormalization", x, axis_i=int(axis), p_i=int(p))


class LocalAttentionModule(nn.Module):
    """LocalAttentionModule.

    Reference: https://github.com/lxtGH/GALD-DGCNet.
    """

    def __init__(
        self,
        num_channels: int,
        normalization: Callable[..., nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.normalization = normalization

        self.dwconv1 = Conv2dModule(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.num_channels,
            normalization=build_norm_layer(self.normalization, num_features=self.num_channels),
            activation=build_activation_layer(nn.ReLU),
        )
        self.dwconv2 = Conv2dModule(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.num_channels,
            normalization=build_norm_layer(self.normalization, num_features=self.num_channels),
            activation=build_activation_layer(nn.ReLU),
        )
        self.dwconv3 = Conv2dModule(
            in_channels=self.num_channels,
            out_channels=self.num_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.num_channels,
            normalization=build_norm_layer(self.normalization, num_features=self.num_channels),
            activation=build_activation_layer(nn.ReLU),
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        _, _, h, w = x.size()

        y = self.dwconv1(x)
        y = self.dwconv2(y)
        y = self.dwconv3(y)
        y = f.interpolate(y, size=(h, w), mode="bilinear", align_corners=True)
        mask = self.sigmoid_spatial(y)

        return x + x * mask
