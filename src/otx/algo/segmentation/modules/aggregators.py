# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Aggregators for semantic segmentation."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.nn import functional as f

from otx.algo.modules import Conv2dModule, DepthwiseSeparableConvModule
from otx.algo.modules.activation import build_activation_layer
from otx.algo.modules.norm import build_norm_layer

from .utils import normalize


class IterativeAggregator(nn.Module):
    """IterativeAggregator.

    Based on: https://github.com/HRNet/Lite-HRNet.

    Args:
        in_channels (list[int]): List of input channels for each branch.
        min_channels (int | None): Minimum number of channels. Defaults to None.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``nn.BatchNorm2d``.
        merge_norm (str | None): Whether to merge normalization layers. Defaults to None.
        use_concat (bool): Whether to use concatenation. Defaults to False.
    """

    def __init__(
        self,
        in_channels: list[int],
        min_channels: int | None = None,
        normalization: Callable[..., nn.Module] | None = nn.BatchNorm2d,
        merge_norm: str | None = None,
        use_concat: bool = False,
    ) -> None:
        """IterativeAggregator for LiteHRNet."""
        super().__init__()

        self.use_concat = use_concat

        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]

        min_channels = min_channels if min_channels is not None else 0

        projects: list[DepthwiseSeparableConvModule | None] = []
        expanders: list[Conv2dModule | None] = []
        fuse_layers: list[Conv2dModule | None] = []

        for i in range(num_branches):
            if not self.use_concat or i == 0:
                fuse_layers.append(None)
            else:
                out_channels = self.in_channels[i + 1]
                fuse_layers.append(
                    Conv2dModule(
                        in_channels=2 * out_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        normalization=build_norm_layer(normalization, num_features=out_channels),
                        activation=build_activation_layer(nn.ReLU),
                    ),
                )

            if i != num_branches - 1:
                out_channels = max(self.in_channels[i + 1], min_channels)
            else:
                out_channels = max(self.in_channels[i], min_channels)

            projects.append(
                DepthwiseSeparableConvModule(
                    in_channels=max(self.in_channels[i], min_channels),
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    normalization=build_norm_layer(normalization, num_features=out_channels),
                    activation=build_activation_layer(nn.ReLU),
                    dw_activation=None,
                    pw_activation=build_activation_layer(nn.ReLU),
                ),
            )

            if self.in_channels[i] < min_channels:
                expanders.append(
                    Conv2dModule(
                        in_channels=self.in_channels[i],
                        out_channels=min_channels,
                        kernel_size=1,
                        stride=1,
                        normalization=build_norm_layer(normalization, num_features=min_channels),
                        activation=build_activation_layer(nn.ReLU),
                    ),
                )
            else:
                expanders.append(None)

        self.projects = nn.ModuleList(projects)
        self.expanders = nn.ModuleList(expanders)
        self.fuse_layers = nn.ModuleList(fuse_layers)

        self.merge_norm = merge_norm

    @staticmethod
    def _norm(x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        """Normalize."""
        if mode is None or mode == "none":
            out = x
        elif mode == "channel":
            out = normalize(x, dim=1, p=2)
        else:
            _, c, h, w = x.size()
            y = x.view(-1, c, h * w)
            y = normalize(y, dim=2, p=2)
            out = y.view(-1, c, h, w)

        return out

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Perform forward pass through the network.

        Args:
        - x (List[Tensor]): Input tensor.

        Returns:
        - List[Tensor]: Output tensor list.
        """
        x = x[::-1]

        y_list = []
        last_x = None
        for i, s_in in enumerate(x):
            s = s_in
            if self.expanders[i] is not None:
                s = self.expanders[i](s)

            if last_x is not None:
                last_x = f.interpolate(last_x, size=s.size()[-2:], mode="bilinear", align_corners=True)

                norm_s = self._norm(s, self.merge_norm)
                norm_x = self._norm(last_x, self.merge_norm)

                if self.use_concat:
                    concat_s = torch.cat([norm_s, norm_x], dim=1)
                    s = self.fuse_layers[i](concat_s)
                else:
                    s = norm_s + norm_x

            s = self.projects[i](s)
            last_x = s

            y_list.append(s)

        return y_list[::-1]
