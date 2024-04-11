# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utils used for LiteHRNet model."""

from __future__ import annotations

from typing import Callable, ClassVar

import torch
import torch.nn.functional as f
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from torch import nn
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (Tensor): The input tensor.
        groups (int): The number of groups to divide the input tensor in the channel dimension.

    Returns:
        Tensor: The output tensor after channel shuffle operation.
    """
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    return x.view(batch_size, -1, height, width)


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
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels if value_channels is not None else in_channels
        self.conv_cfg = conv_cfg
        if norm_cfg is None:
            norm_cfg = {"type": "BN"}
        if psp_size is None:
            psp_size = (1, 3, 6, 8)
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


def normalize(x: torch.Tensor, dim: int, p: int = 2, eps: float = 1e-12) -> torch.Tensor:
    """Normalize method."""
    if torch.onnx.is_in_onnx_export():
        return OnnxLpNormalization.apply(x, dim, p, eps)
    return torch.nn.functional.normalize(x, dim=dim, p=p, eps=eps)


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
        super().__init__()

        self.use_concat = use_concat

        num_branches = len(in_channels)
        self.in_channels = in_channels[::-1]

        min_channels = min_channels if min_channels is not None else 0

        projects, expanders = [], []
        fuse_layers: list[ConvModule] = []
        for i in range(num_branches):
            if not self.use_concat or i == 0:
                fuse_layers.append(None)
            else:
                out_channels = self.in_channels[i + 1]
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


class LocalAttentionModule(nn.Module):
    """LocalAttentionModule.

    Reference: https://github.com/lxtGH/GALD-DGCNet.
    """

    def __init__(self, num_channels: int, conv_cfg: dict | None = None, norm_cfg: dict | None = None):
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
        y = f.interpolate(y, size=(h, w), mode="bilinear", align_corners=True)
        mask = self.sigmoid_spatial(y)

        return x + x * mask
