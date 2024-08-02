# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Aggregators for semantic segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as f

from otx.algo.modules import Conv2dModule, DepthwiseSeparableConvModule

from .utils import normalize


class IterativeAggregator(nn.Module):
    """IterativeAggregator.

    Based on: https://github.com/HRNet/Lite-HRNet.
    """

    def __init__(
        self,
        in_channels: list[int],
        min_channels: int | None = None,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
        merge_norm: str | None = None,
        use_concat: bool = False,
    ) -> None:
        """IterativeAggregator for LiteHRNet.

        Args:
            in_channels (list[int]): List of input channels for each branch.
            min_channels (int | None): Minimum number of channels. Defaults to None.
            conv_cfg (dict | None): Config for convolution layers. Defaults to None.
            norm_cfg (dict | None): Config for normalization layers. Defaults to None.
            merge_norm (str | None): Whether to merge normalization layers. Defaults to None.
            use_concat (bool): Whether to use concatenation. Defaults to False.

        Returns:
            None
        """
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        if conv_cfg is None:
            conv_cfg = {"type": "Conv2d"}

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
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg={"type": "ReLU"},
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
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg={"type": "ReLU"},
                    dw_act_cfg=None,
                    pw_act_cfg={"type": "ReLU"},
                ),
            )

            if self.in_channels[i] < min_channels:
                expanders.append(
                    Conv2dModule(
                        in_channels=self.in_channels[i],
                        out_channels=min_channels,
                        kernel_size=1,
                        stride=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg={"type": "ReLU"},
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
