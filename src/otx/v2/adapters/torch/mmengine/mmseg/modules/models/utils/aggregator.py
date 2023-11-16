"""Aggregators."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from torch import nn

from otx.v2.adapters.torch.mmengine.mmseg.modules.models.utils.normalize import normalize


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
        """Initializes IterativeAggregator.

        Args:
            in_channels (list[int]): Number of input channels for each branch.
            min_channels (int, optional): Minimum number of output channels. Defaults to None.
            conv_cfg (dict, optional): Config for convolution layers. Defaults to None.
            norm_cfg (dict, optional): Config for normalization layers. Defaults to None.
            merge_norm (str, optional): Type of normalization to apply after feature aggregation. Allowed values are
                None, 'none', 'channel', 'spatial'. Defaults to None.
            use_concat (bool, optional): Whether to concatenate features from all branches. Defaults to False.

        Raises:
            ValueError: If `merge_norm` is not None, 'none', 'channel', or 'spatial'.
        """
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        super().__init__()

        self.use_concat = use_concat

        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]

        min_channels = min_channels if min_channels is not None else 0
        if min_channels < 0:
            msg = f"min_channels must be greater than or equal to 0. Got {min_channels}."
            raise ValueError(msg)

        out_channels = 1
        projects, expanders = [], []
        fuse_layers: list[torch.nn.Module] = []
        for i in range(num_branches):
            if not self.use_concat or i == 0:
                fuse_layers.append(None)
            else:
                fuse_layers.append(
                    ConvModule(
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
                    ConvModule(
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

        if merge_norm not in [None, "none", "channel", "spatial"]:
            msg = "Invalid value for merge_norm. Allowed values are None, 'none', 'channel', 'spatial'."
            raise ValueError(msg)
        self.merge_norm = merge_norm

    @staticmethod
    def _norm(x: torch.Tensor, mode: str | None = None) -> torch.Tensor:
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
        """Forward."""
        x = x[::-1]

        y_list: list[torch.Tensor] = []
        last_x = None
        for i, s in enumerate(x):
            if self.expanders[i] is not None:
                s = self.expanders[i](s)  # noqa: PLW2901

            if last_x is not None:
                last_x = torch.nn.functional.interpolate(
                    last_x,
                    size=s.size()[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

                norm_s = self._norm(s, self.merge_norm)
                norm_x = self._norm(last_x, self.merge_norm)

                if self.use_concat:
                    concat_s = torch.cat([norm_s, norm_x], dim=1)
                    s = self.fuse_layers[i](concat_s)  # noqa: PLW2901
                else:
                    s = norm_s + norm_x  # noqa: PLW2901

            s = self.projects[i](s)  # noqa: PLW2901
            last_x = s

            y_list.append(s)

        return y_list[::-1]
