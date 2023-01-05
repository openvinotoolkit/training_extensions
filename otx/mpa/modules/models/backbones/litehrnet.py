# Copyright (c) 2018-2020 Open-MMLab.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2021 DeLightCMU
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Modified from: https://github.com/HRNet/Lite-HRNet"""


import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import (
    ConvModule,
    build_conv_layer,
    build_norm_layer,
    constant_init,
    normal_init,
)
from mmcv.runner import BaseModule, load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.models.backbones.resnet import BasicBlock, Bottleneck
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger

from ..utils import (
    AsymmetricPositionAttentionModule,
    IterativeAggregator,
    LocalAttentionModule,
    channel_shuffle,
)


class NeighbourSupport(nn.Module):
    def __init__(self, channels, kernel_size=3, key_ratio=8, value_ratio=8, conv_cfg=None, norm_cfg=None):
        super().__init__()

        self.in_channels = channels
        self.key_channels = int(channels / key_ratio)
        self.value_channels = int(channels / value_ratio)
        self.kernel_size = kernel_size

        self.key = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=self.key_channels,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type="ReLU"),
            ),
            ConvModule(
                self.key_channels,
                self.key_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding=(self.kernel_size - 1) // 2,
                groups=self.key_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ),
            ConvModule(
                in_channels=self.key_channels,
                out_channels=self.kernel_size * self.kernel_size,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ),
        )
        self.value = nn.Sequential(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=self.value_channels,
                kernel_size=1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ),
            nn.Unfold(kernel_size=self.kernel_size, stride=1, padding=1),
        )
        self.out_conv = ConvModule(
            in_channels=self.value_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    def forward(self, x):
        h, w = [int(_) for _ in x.size()[-2:]]

        key = self.key(x).view(-1, 1, self.kernel_size**2, h, w)
        weights = torch.softmax(key, dim=2)

        value = self.value(x).view(-1, self.value_channels, self.kernel_size**2, h, w)
        y = torch.sum(weights * value, dim=2)
        y = self.out_conv(y)

        out = x + y

        return out


class CrossResolutionWeighting(nn.Module):
    def __init__(
        self, channels, ratio=16, conv_cfg=None, norm_cfg=None, act_cfg=(dict(type="ReLU"), dict(type="Sigmoid"))
    ):
        super().__init__()

        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)

        self.channels = channels
        total_channel = sum(channels)

        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0],
        )
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1],
        )

    def forward(self, x):
        min_size = [int(_) for _ in x[-1].size()[-2:]]

        out = [F.adaptive_avg_pool2d(s, min_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [s * F.interpolate(a, size=s.size()[-2:], mode="nearest") for s, a in zip(x, out)]

        return out


class SpatialWeighting(nn.Module):
    def __init__(self, channels, ratio=16, conv_cfg=None, act_cfg=(dict(type="ReLU"), dict(type="Sigmoid")), **kwargs):
        super().__init__()

        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0],
        )
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1],
        )

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)

        return x * out


class SpatialWeightingV2(nn.Module):
    """The original repo: https://github.com/DeLightCMU/PSA"""

    def __init__(self, channels, ratio=16, conv_cfg=None, norm_cfg=None, enable_norm=False, **kwargs):
        super().__init__()

        self.in_channels = channels
        self.internal_channels = int(channels / ratio)

        # channel-only branch
        self.v_channel = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.internal_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if enable_norm else None,
            act_cfg=None,
        )
        self.q_channel = ConvModule(
            in_channels=self.in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if enable_norm else None,
            act_cfg=None,
        )
        self.out_channel = ConvModule(
            in_channels=self.internal_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="Sigmoid"),
        )

        # spatial-only branch
        self.v_spatial = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.internal_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if enable_norm else None,
            act_cfg=None,
        )
        self.q_spatial = ConvModule(
            in_channels=self.in_channels,
            out_channels=self.internal_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg if enable_norm else None,
            act_cfg=None,
        )
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

    def _channel_weighting(self, x):
        h, w = [int(_) for _ in x.size()[-2:]]

        v = self.v_channel(x).view(-1, self.internal_channels, h * w)

        q = self.q_channel(x).view(-1, h * w, 1)
        q = torch.softmax(q, dim=1)

        y = torch.matmul(v, q)
        y = y.view(-1, self.internal_channels, 1, 1)
        y = self.out_channel(y)

        out = x * y

        return out

    def _spatial_weighting(self, x):
        h, w = [int(_) for _ in x.size()[-2:]]

        v = self.v_spatial(x)
        v = v.view(-1, self.internal_channels, h * w)

        q = self.q_spatial(x)
        q = self.global_avgpool(q)
        q = torch.softmax(q, dim=1)
        q = q.view(-1, 1, self.internal_channels)

        y = torch.matmul(q, v)
        y = y.view(-1, 1, h, w)
        y = torch.sigmoid(y)

        out = x * y

        return out

    def forward(self, x):
        y_channel = self._channel_weighting(x)
        y_spatial = self._spatial_weighting(x)
        out = y_channel + y_spatial

        return out


class ConditionalChannelWeighting(nn.Module):
    def __init__(
        self,
        in_channels,
        stride,
        reduce_ratio,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
        dropout=None,
        weighting_module_version="v1",
        neighbour_weighting=False,
        dw_ksize=3,
    ):
        super().__init__()

        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        spatial_weighting_module = SpatialWeighting if weighting_module_version == "v1" else SpatialWeightingV2
        branch_channels = [channel // 2 for channel in in_channels]

        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels, ratio=reduce_ratio, conv_cfg=conv_cfg, norm_cfg=norm_cfg
        )
        self.depthwise_convs = nn.ModuleList(
            [
                ConvModule(
                    channel,
                    channel,
                    kernel_size=dw_ksize,
                    stride=self.stride,
                    padding=dw_ksize // 2,
                    groups=channel,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                )
                for channel in branch_channels
            ]
        )
        self.spatial_weighting = nn.ModuleList(
            [
                spatial_weighting_module(
                    channels=channel, ratio=4, conv_cfg=conv_cfg, norm_cfg=norm_cfg, enable_norm=True
                )
                for channel in branch_channels
            ]
        )

        self.neighbour_weighting = None
        if neighbour_weighting:
            self.neighbour_weighting = nn.ModuleList(
                [
                    NeighbourSupport(
                        channel, kernel_size=3, key_ratio=8, value_ratio=4, conv_cfg=conv_cfg, norm_cfg=norm_cfg
                    )
                    for channel in branch_channels
                ]
            )

        self.dropout = None
        if dropout is not None and dropout > 0:
            self.dropout = nn.ModuleList([nn.Dropout(p=dropout) for _ in branch_channels])

    def _inner_forward(self, x):
        x = [s.chunk(2, dim=1) for s in x]
        x1 = [s[0] for s in x]
        x2 = [s[1] for s in x]

        x2 = self.cross_resolution_weighting(x2)
        x2 = [dw(s) for s, dw in zip(x2, self.depthwise_convs)]

        if self.neighbour_weighting is not None:
            x2 = [nw(s) for s, nw in zip(x2, self.neighbour_weighting)]

        x2 = [sw(s) for s, sw in zip(x2, self.spatial_weighting)]

        if self.dropout is not None:
            x2 = [dropout(s) for s, dropout in zip(x2, self.dropout)]

        out = [torch.cat([s1, s2], dim=1) for s1, s2 in zip(x1, x2)]
        out = [channel_shuffle(s, 2) for s in out]

        return out

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self._inner_forward, x)
        else:
            out = self._inner_forward(x)

        return out


class Stem(nn.Module):
    def __init__(
        self,
        in_channels,
        stem_channels,
        out_channels,
        expand_ratio,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
        strides=(2, 2),
        extra_stride=False,
        input_norm=False,
    ):
        super().__init__()

        assert isinstance(strides, (tuple, list))
        assert len(strides) == 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

        self.input_norm = None
        if input_norm:
            self.input_norm = nn.InstanceNorm2d(in_channels)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=strides[0],
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type="ReLU"),
        )

        self.conv2 = None
        if extra_stride:
            self.conv2 = ConvModule(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type="ReLU"),
            )

        mid_channels = int(round(stem_channels * expand_ratio))
        branch_channels = stem_channels // 2
        if stem_channels == self.out_channels:
            inc_channels = self.out_channels - branch_channels
        else:
            inc_channels = self.out_channels - stem_channels

        self.branch1 = nn.Sequential(
            ConvModule(
                branch_channels,
                branch_channels,
                kernel_size=3,
                stride=strides[1],
                padding=1,
                groups=branch_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ),
            ConvModule(
                branch_channels,
                inc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=dict(type="ReLU"),
            ),
        )

        self.expand_conv = ConvModule(
            branch_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="ReLU"),
        )
        self.depthwise_conv = ConvModule(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=strides[1],
            padding=1,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )
        self.linear_conv = ConvModule(
            mid_channels,
            branch_channels if stem_channels == self.out_channels else stem_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="ReLU"),
        )

    def _inner_forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)

        x = self.conv1(x)
        if self.conv2 is not None:
            x = self.conv2(x)

        x1, x2 = x.chunk(2, dim=1)

        x1 = self.branch1(x1)

        x2 = self.expand_conv(x2)
        x2 = self.depthwise_conv(x2)
        x2 = self.linear_conv(x2)

        out = torch.cat((x1, x2), dim=1)
        out = channel_shuffle(out, 2)

        return out

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self._inner_forward, x)
        else:
            out = self._inner_forward(x)

        return out


class StemV2(nn.Module):
    def __init__(
        self,
        in_channels,
        stem_channels,
        out_channels,
        expand_ratio,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
        num_stages=1,
        strides=(2, 2),
        extra_stride=False,
        input_norm=False,
    ):
        super().__init__()

        assert num_stages > 0
        assert isinstance(strides, (tuple, list))
        assert len(strides) == 1 + num_stages

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.num_stages = num_stages

        self.input_norm = None
        if input_norm:
            self.input_norm = nn.InstanceNorm2d(in_channels)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=strides[0],
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type="ReLU"),
        )

        self.conv2 = None
        if extra_stride:
            self.conv2 = ConvModule(
                in_channels=stem_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type="ReLU"),
            )

        mid_channels = int(round(stem_channels * expand_ratio))
        internal_branch_channels = stem_channels // 2
        out_branch_channels = self.out_channels // 2

        self.branch1, self.branch2 = nn.ModuleList(), nn.ModuleList()
        for stage in range(1, num_stages + 1):
            self.branch1.append(
                nn.Sequential(
                    ConvModule(
                        internal_branch_channels,
                        internal_branch_channels,
                        kernel_size=3,
                        stride=strides[stage],
                        padding=1,
                        groups=internal_branch_channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None,
                    ),
                    ConvModule(
                        internal_branch_channels,
                        out_branch_channels if stage == num_stages else internal_branch_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type="ReLU"),
                    ),
                )
            )

            self.branch2.append(
                nn.Sequential(
                    ConvModule(
                        internal_branch_channels,
                        mid_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type="ReLU"),
                    ),
                    ConvModule(
                        mid_channels,
                        mid_channels,
                        kernel_size=3,
                        stride=strides[stage],
                        padding=1,
                        groups=mid_channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=None,
                    ),
                    ConvModule(
                        mid_channels,
                        out_branch_channels if stage == num_stages else internal_branch_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type="ReLU"),
                    ),
                )
            )

    def _inner_forward(self, x):
        if self.input_norm is not None:
            x = self.input_norm(x)

        y = self.conv1(x)
        if self.conv2 is not None:
            y = self.conv2(y)

        out_list = [y]
        for stage in range(self.num_stages):
            y1, y2 = y.chunk(2, dim=1)

            y1 = self.branch1[stage](y1)
            y2 = self.branch2[stage](y2)

            y = torch.cat((y1, y2), dim=1)
            y = channel_shuffle(y, 2)
            out_list.append(y)

        return out_list

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self._inner_forward, x)
        else:
            out = self._inner_forward(x)

        return out


class ShuffleUnit(nn.Module):
    """InvertedResidual block for ShuffleNetV2 backbone.

    Args:
        in_channels (int): The input channels of the block.
        out_channels (int): The output channels of the block.
        stride (int): Stride of the 3x3 convolution layer. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        with_cp=False,
    ):
        super().__init__()
        self.stride = stride
        self.with_cp = with_cp

        branch_features = out_channels // 2
        if self.stride == 1:
            assert in_channels == branch_features * 2, (
                f"in_channels ({in_channels}) should equal to "
                f"branch_features * 2 ({branch_features * 2}) "
                "when stride is 1"
            )

        if in_channels != branch_features * 2:
            assert self.stride != 1, (
                f"stride ({self.stride}) should not equal 1 when " f"in_channels != branch_features * 2"
            )

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=in_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                ),
                ConvModule(
                    in_channels,
                    branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                ),
            )

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                groups=branch_features,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            ),
            ConvModule(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

    def _inner_forward(self, x):
        if self.stride > 1:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)

        out = channel_shuffle(out, 2)

        return out

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self._inner_forward, x)
        else:
            out = self._inner_forward(x)

        return out


class LiteHRModule(nn.Module):
    def __init__(
        self,
        num_branches,
        num_blocks,
        in_channels,
        reduce_ratio,
        module_type,
        multiscale_output=False,
        with_fuse=True,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        with_cp=False,
        dropout=None,
        weighting_module_version="v1",
        neighbour_weighting=False,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.weighting_module_version = weighting_module_version
        self.neighbour_weighting = neighbour_weighting

        if self.module_type == "LITE":
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio, dropout=dropout)
        elif self.module_type == "NAIVE":
            self.layers = self._make_naive_branches(num_branches, num_blocks)

        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    @staticmethod
    def _check_branches(num_branches, in_channels):
        """Check input to avoid ValueError."""

        if num_branches != len(in_channels):
            error_msg = f"NUM_BRANCHES({num_branches}) != NUM_INCHANNELS({len(in_channels)})"
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1, dropout=None):
        layers = []
        for i in range(num_blocks):
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp,
                    dropout=dropout,
                    weighting_module_version=self.weighting_module_version,
                    neighbour_weighting=self.neighbour_weighting,
                )
            )

        return nn.Sequential(*layers)

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        """Make one branch."""

        layers = [
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=stride,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type="ReLU"),
                with_cp=self.with_cp,
            )
        ]
        for i in range(1, num_blocks):
            layers.append(
                ShuffleUnit(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    stride=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=dict(type="ReLU"),
                    with_cp=self.with_cp,
                )
            )

        return nn.Sequential(*layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        """Make branches."""

        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""

        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        num_out_branches = num_branches if self.multiscale_output else 1

        fuse_layers = []
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[i])[1],
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False,
                                    ),
                                    build_norm_layer(self.norm_cfg, in_channels[j])[1],
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""

        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        if self.module_type == "LITE":
            out = self.layers(x)
        elif self.module_type == "NAIVE":
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            out = x

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        fuse_y = out[j]
                    else:
                        fuse_y = self.fuse_layers[i][j](out[j])

                    if fuse_y.size()[-2:] != y.size()[-2:]:
                        fuse_y = F.interpolate(fuse_y, size=y.size()[-2:], mode="nearest")

                    y += fuse_y

                out_fuse.append(self.relu(y))

            out = out_fuse
        elif not self.multiscale_output:
            out = [out[0]]

        return out


@BACKBONES.register_module()
class LiteHRNet(BaseModule):
    """Lite-HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    <https://arxiv.org/abs/1904.04514>`_

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """

    def __init__(
        self,
        extra,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=dict(type="BN"),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
        dropout=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.stem = Stem(
            in_channels,
            input_norm=self.extra["stem"]["input_norm"],
            stem_channels=self.extra["stem"]["stem_channels"],
            out_channels=self.extra["stem"]["out_channels"],
            expand_ratio=self.extra["stem"]["expand_ratio"],
            strides=self.extra["stem"]["strides"],
            extra_stride=self.extra["stem"]["extra_stride"],
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
        )

        self.enable_stem_pool = self.extra["stem"].get("out_pool", False)
        if self.enable_stem_pool:
            self.stem_pool = nn.AvgPool2d(kernel_size=3, stride=2)

        self.num_stages = self.extra["num_stages"]
        self.stages_spec = self.extra["stages_spec"]

        num_channels_last = [
            self.stem.out_channels,
        ]
        for i in range(self.num_stages):
            num_channels = self.stages_spec["num_channels"][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]

            setattr(self, "transition{}".format(i), self._make_transition_layer(num_channels_last, num_channels))

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, multiscale_output=True, dropout=dropout
            )
            setattr(self, "stage{}".format(i), stage)

        self.out_modules = None
        if self.extra.get("out_modules") is not None:
            out_modules = []
            in_modules_channels, out_modules_channels = num_channels_last[-1], None
            if self.extra["out_modules"]["conv"]["enable"]:
                out_modules_channels = self.extra["out_modules"]["conv"]["channels"]
                out_modules.append(
                    ConvModule(
                        in_channels=in_modules_channels,
                        out_channels=out_modules_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=dict(type="ReLU"),
                    )
                )
                in_modules_channels = out_modules_channels
            if self.extra["out_modules"]["position_att"]["enable"]:
                out_modules.append(
                    AsymmetricPositionAttentionModule(
                        in_channels=in_modules_channels,
                        key_channels=self.extra["out_modules"]["position_att"]["key_channels"],
                        value_channels=self.extra["out_modules"]["position_att"]["value_channels"],
                        psp_size=self.extra["out_modules"]["position_att"]["psp_size"],
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                    )
                )
            if self.extra["out_modules"]["local_att"]["enable"]:
                out_modules.append(
                    LocalAttentionModule(
                        num_channels=in_modules_channels,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                    )
                )

            if len(out_modules) > 0:
                self.out_modules = nn.Sequential(*out_modules)
                num_channels_last.append(in_modules_channels)

        self.add_stem_features = self.extra.get("add_stem_features", False)
        if self.add_stem_features:
            self.stem_transition = nn.Sequential(
                ConvModule(
                    self.stem.out_channels,
                    self.stem.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=self.stem.out_channels,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                ),
                ConvModule(
                    self.stem.out_channels,
                    num_channels_last[0],
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="ReLU"),
                ),
            )

            num_channels_last = [num_channels_last[0]] + num_channels_last

        self.with_aggregator = self.extra.get("out_aggregator") and self.extra["out_aggregator"]["enable"]
        if self.with_aggregator:
            self.aggregator = IterativeAggregator(
                in_channels=num_channels_last,
                min_channels=self.extra["out_aggregator"].get("min_channels", None),
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
            )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """Make transition layer."""

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, num_channels_pre_layer[i])[1],
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, num_channels_cur_layer[i])[1],
                            nn.ReLU(),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg, in_channels)[1],
                            build_conv_layer(
                                self.conv_cfg, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
                            ),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, stages_spec, stage_index, in_channels, multiscale_output=True, dropout=None):
        num_modules = stages_spec["num_modules"][stage_index]
        num_branches = stages_spec["num_branches"][stage_index]
        num_blocks = stages_spec["num_blocks"][stage_index]
        reduce_ratio = stages_spec["reduce_ratios"][stage_index]
        with_fuse = stages_spec["with_fuse"][stage_index]
        module_type = stages_spec["module_type"][stage_index]
        weighting_module_version = stages_spec.get("weighting_module_version", "v1")
        neighbour_weighting = stages_spec.get("neighbour_weighting", False)

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                LiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp,
                    dropout=dropout,
                    weighting_module_version=weighting_module_version,
                    neighbour_weighting=neighbour_weighting,
                )
            )
            in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        """Forward function."""

        stem_outputs = self.stem(x)
        y_x2 = y_x4 = stem_outputs
        # y_x2, y_x4 = stem_outputs[-2:]
        y = y_x4

        if self.enable_stem_pool:
            y = self.stem_pool(y)

        y_list = [y]
        for i in range(self.num_stages):
            transition_modules = getattr(self, "transition{}".format(i))

            stage_inputs = []
            for j in range(self.stages_spec["num_branches"][i]):
                if transition_modules[j]:
                    if j >= len(y_list):
                        stage_inputs.append(transition_modules[j](y_list[-1]))
                    else:
                        stage_inputs.append(transition_modules[j](y_list[j]))
                else:
                    stage_inputs.append(y_list[j])

            stage_module = getattr(self, "stage{}".format(i))
            y_list = stage_module(stage_inputs)

        if self.out_modules is not None:
            y_list.append(self.out_modules(y_list[-1]))

        if self.add_stem_features:
            y_stem = self.stem_transition(y_x2)
            y_list = [y_stem] + y_list

        out = y_list
        if self.with_aggregator:
            out = self.aggregator(out)

        if self.extra.get("add_input", False):
            out = [x] + out

        return out

    def train(self, mode=True):
        """Convert the model into training mode."""

        super().train(mode)

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
