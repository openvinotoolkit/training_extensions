# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks/modules for semantic segmentation"""

from __future__ import annotations

from typing import Callable, ClassVar

import torch
import torch.nn.functional as f
from torch import nn
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d

from otx.algo.modules import ConvModule


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


# class myLayerNorm(nn.Module):
#     def __init__(self, inChannels):
#         super().__init__()
#         self.norm == nn.LayerNorm(inChannels, eps=1e-5)

#     def forward(self, x):
#         # reshaping only to apply Layer Normalization layer
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
#         x = self.norm(x)
#         x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # B*HW*C -> B*H*W*C -> B*C*H*W

#         return x


# class NormLayer(nn.Module):
#     #def __init__(self, inChannels, norm_type=config['norm_typ']):
#     def __init__(self, inChannels, norm_type=dict(type='SyncBN', requires_grad=True), momentum=3e-4):
#         super().__init__()
#         self.inChannels = inChannels
#         self.norm_type = norm_type
#         if norm_type['type'] == 'BN':
#             # print('Adding Batch Norm layer') # for testing
#             self.norm = nn.BatchNorm2d(inChannels, eps=1e-5, momentum=momentum)
#         elif norm_type['type'] == 'SyncBN':
#             # print('Adding Sync-Batch Norm layer') # for testing
#             self.norm = norm_layer(inChannels, momentum=momentum)
#         elif norm_type['type'] == 'layer_norm':
#             # print('Adding Layer Norm layer') # for testing
#             self.norm == nn.myLayerNorm(inChannels)
#         else:
#             raise NotImplementedError


#     def build_norm_layer(in_channels, norm_type=dict(type='SyncBN', requires_grad=True), momentum=3e-4):
#         if norm_type['type'] == 'BN':
#             return nn.BatchNorm2d(in_channels, eps=1e-5, momentum=momentum)
#         elif norm_type['type'] == 'SyncBN':
#             return norm_layer(in_channels, momentum=momentum)
#         elif norm_type['type'] == 'layer_norm':
#             return nn.myLayerNorm(in_channels)
#         else:
#             raise NotImplementedError


#     def forward(self, x):

#         x = self.norm(x)

#         return x

#     def __repr__(self):
#         return f'{self.__class__.__name__}(dim={self.inChannels}, norm_type={self.norm_type})'


# class LayerScale(nn.Module):
#     '''
#     Layer scale module.
#     References:
#       - https://arxiv.org/abs/2103.17239
#     '''
#     def __init__(self, inChannels, init_value=1e-2):
#         super().__init__()
#         self.inChannels = inChannels
#         self.init_value = init_value
#         self.layer_scale = nn.Parameter(init_value * torch.ones((inChannels)), requires_grad=True)

#     def forward(self, x):
#         if self.init_value == 0.0:
#             return x
#         else:
#             scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1) # C, -> C,1,1
#             return scale * x

#     def __repr__(self):
#         return f'{self.__class__.__name__}(dim={self.inChannels}, init_value={self.init_value})'


# def stochastic_depth(input: torch.Tensor, p: float,
#                      mode: str, training: bool =  True):

#     if not training or p == 0.0:
#         # print(f'not adding stochastic depth of: {p}')
#         return input

#     survival_rate = 1.0 - p
#     if mode == 'row':
#         shape = [input.shape[0]] + [1] * (input.ndim - 1) # just converts BXCXHXW -> [B,1,1,1] list
#     elif mode == 'batch':
#         shape = [1] * input.ndim

#     noise = torch.empty(shape, dtype=input.dtype, device=input.device)
#     noise = noise.bernoulli_(survival_rate)
#     if survival_rate > 0.0:
#         noise.div_(survival_rate)
#     # print(f'added sDepth of: {p}')
#     return input * noise

# class StochasticDepth(nn.Module):
#     '''
#     Stochastic Depth module.
#     It performs ROW-wise dropping rather than sample-wise.
#     mode (str): ``"batch"`` or ``"row"``.
#                 ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
#                 randomly selected rows from the batch.
#     References:
#       - https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
#     '''
#     def __init__(self, p=0.5, mode='row'):
#         super().__init__()
#         self.p = p
#         self.mode = mode

#     def forward(self, input):
#         return stochastic_depth(input, self.p, self.mode, self.training)

#     def __repr__(self):
#        s = f"{self.__class__.__name__}(p={self.p})"
#        return s


# class DownSample(nn.Module):
#     def __init__(self, kernelSize=3, stride=2, in_channels=3, embed_dim=768):
#         super().__init__()
#         self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(kernelSize, kernelSize),
#                               stride=stride, padding=(kernelSize//2, kernelSize//2))
#         # stride 4 => 4x down sample
#         # stride 2 => 2x down sample
#     def forward(self, x):

#         x = self.proj(x)
#         B, C, H, W = x.size()
#         # x = x.flatten(2).transpose(1,2)
#         return x, H, W


# class DWConv3x3(nn.Module):
#     '''Depth wise conv'''
#     def __init__(self, dim=768):
#         super(DWConv3x3, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

#     def forward(self, x):
#         x = self.dwconv(x)
#         return x

# class ConvBNRelu(nn.Module):

#     @classmethod
#     def _same_paddings(cls, kernel):
#         if kernel == 1:
#             return 0
#         elif kernel == 3:
#             return 1

#     def __init__(self, inChannels, outChannels, kernel=3, stride=1, padding='same',
#                  dilation=1, groups=1, norm_type='batch_norm'):
#         super().__init__()

#         if padding == 'same':
#             padding = self._same_paddings(kernel)

#         self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel,
#                               padding=padding, stride=stride, dilation=dilation,
#                               groups=groups, bias=False)
#         self.norm = NormLayer(outChannels, norm_type=norm_type)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):

#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act(x)

#         return x

# class DWSeprableConvModule(nn.Module):
#     def __init__(self, inChannels, outChannels, kernal_size=3, stride=1, padding=1, dialation=1,
#                  norm_cfg=None, act_cfg=None, dw_norm_cfg=None, dw_act_cfg=None,
#                  pw_norm_cfg=None, pw_act_cfg=None, bias=False):
#         self.dwconv = ConvModule(inChannels, inChannels, kernal_size=kernal_size, stride=stride, padding=padding,
#                                 dialation=dialation, groups=inChannels, norm_cfg=dw_norm_cfg, act_cfg=dw_act_cfg, bias=bias)
#         self.pwconv = ConvModule(inChannels, outChannels, kernal_size=1, norm_cfg=pw_norm_cfg, act_cfg=pw_act_cfg, bias=bias)

#     def forward(self, x):

#         x = self.dwconv(x)
#         x = self.pwconv(x)

#         return x

# class ConvRelu(nn.Module):
#     def __init__(self, inChannels, outChannels, kernel=1, bias=False):
#         super().__init__()
#         self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel, bias=False)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):

#         x = self.conv(x)
#         x = self.act(x)

#         return x
