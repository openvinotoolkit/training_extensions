# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.layers.transformer.utils.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/layers/transformer/utils.py
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn.functional
from timm.models.layers import to_2tuple
from torch import nn

from otx.algo.modules.base_module import BaseModule
from otx.algo.modules.norm import build_norm_layer


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed).

    so that input can get fully covered by filter you specified. It support two modes "same" and "corner".
    The "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around input.
    The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    """

    def __init__(
        self,
        kernel_size: int | tuple = 1,
        stride: int | tuple | None = 1,
        dilation: int | tuple = 1,
        padding: str = "corner",
    ) -> None:
        super().__init__()

        if padding not in ("same", "corner"):
            msg = f"padding mode only support 'same' and 'corner', but got {padding}"
            raise ValueError(msg)

        self.padding = to_2tuple(padding)
        self.kernel_size = to_2tuple(kernel_size)
        self.stride = to_2tuple(stride)
        self.dilation = to_2tuple(dilation)

    def get_pad_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        """Get the padding shape."""
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h: int = math.ceil(input_h / stride_h)
        output_w: int = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for AdaptivePadding."""
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = torch.nn.functional.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == "same":
                x = torch.nn.functional.pad(
                    x,
                    [
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    ],
                )
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    TODO (sungchul): it is duplicated with otx.algo.modules.transformer.PatchEmbed

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: None (Would be set as `kernel_size`).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
        init_cfg (dict, optional): The Config for
            initialization. Default: None.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        kernel_size: int = 16,
        stride: int = 16,
        padding: int | tuple | str = "corner",
        dilation: int = 1,
        bias: bool = True,
        normalization: Callable[..., nn.Module] | None = None,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.adap_padding: nn.Module | None
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if normalization is not None:
            self.norm = build_norm_layer(normalization, embed_dims)[1]
        else:
            self.norm = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Forward function for PatchEmbed.

        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """
        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(BaseModule):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    Args:
        in_channels (int): The num of input channels.
            to gets fully covered by filter and stride you specified..
            Default: True.
        out_channels (int): The num of output channels.
        kernel_size (int | tuple, optional): the kernel size in the unfold
            layer. Defaults to 2.
        stride (int | tuple, optional): the stride of the sliding blocks in the
            unfold layer. Default: None. (Would be set as `kernel_size`)
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int | tuple, optional): dilation parameter in the unfold
            layer. Default: 1.
        bias (bool, optional): Whether to add bias in linear layer or not.
            Defaults: False.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to ``nn.LayerNorm``.
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 2,
        stride: int | tuple | None = None,
        padding: int | tuple | str = "corner",
        dilation: int | tuple = 1,
        bias: bool = False,
        normalization: Callable[..., nn.Module] | None = nn.LayerNorm,
        init_cfg: dict | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = stride if stride else kernel_size

        _kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.adap_padding: nn.Module | None
        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=_kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(kernel_size=_kernel_size, dilation=dilation, padding=padding, stride=stride)

        sample_dim = _kernel_size[0] * _kernel_size[1] * in_channels

        if normalization is not None:
            self.norm = build_norm_layer(normalization, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, input_size: tuple[int, ...]) -> tuple[torch.Tensor, tuple[int, int]]:
        """Forward function for PatchMerging.

        Args:
            x (Tensor): Has shape (B, H*W, C_in).
            input_size (tuple[int]): The spatial shape of x, arrange as (H, W).
                Default: None.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, Merged_H * Merged_W, C_out)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (Merged_H, Merged_W).
        """
        batch_size, length, channels = x.shape
        if not isinstance(input_size, Sequence):
            msg = f"Expect input_size is `Sequence` but get {input_size}"
            raise TypeError(msg)

        h, w = input_size
        if h * w != length:
            msg = "input feature has wrong size"
            raise ValueError(msg)

        x = x.view(batch_size, h, w, channels).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            h, w = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (
            h + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            w + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1
        ) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size
