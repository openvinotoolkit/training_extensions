# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.

"""This implementation replaces the functionality of mmcv.cnn.bricks.transformer."""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import torch
from torch import nn

from otx.algo.modules.base_module import BaseModule, Sequential

from .drop import build_dropout
from .norm import build_norm_layer


class AdaptivePadding(nn.Module):
    """Applies padding adaptively to the input.

    This module can make input get fully covered by filter
    you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad
    zero around input. The "corner"  mode would pad zero
    to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel. Default: 1.
        stride (int | tuple): Stride of the filter. Default: 1.
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".

    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding: str = "corner",
    ):
        super().__init__()
        assert padding in ("same", "corner")  # noqa: S101

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        """Calculate the padding size of input.

        Args:
            input_shape (:obj:`torch.Size`): arrange as (H, W).

        Returns:
            Tuple[int]: The padding size along the
            original H and W directions
        """
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add padding to `x`.

        Args:
            x (Tensor): Input tensor has shape (B, C, H, W).

        Returns:
            Tensor: The tensor with adaptive padding
        """
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = nn.functional.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == "same":
                x = nn.functional.pad(
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

    TODO (sungchul): it is duplicated with otx.algo.instance_segmentation.layers.transformer.PatchEmbed

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int): The slide stride of embedding conv.
            Default: 16.
        padding (int | tuple | string): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        normalization (Callable[..., nn.Module] | None): Normalization layer module.
            Defaults to None.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only works when `dynamic_size`
            is False. Default: None.
        init_cfg (`mmcv.ConfigDict`, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: int = 768,
        kernel_size: int | tuple[int, int] = 16,
        stride: int | tuple[int, int] = 16,
        padding: str | int | tuple[int, int] = "corner",
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
        normalization: Callable[..., nn.Module] | None = None,
        input_size: int | tuple[int, int] | None = None,
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        stride = stride if isinstance(stride, tuple) else (stride, stride)
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        self.adaptive_padding: AdaptivePadding | None
        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        self.norm: nn.Module | None
        if normalization is not None:
            self.norm = build_norm_layer(normalization, embed_dims)[1]
        else:
            self.norm = None

        self.init_input_size: tuple[int, int]
        self.init_out_size: tuple[int, int]
        if input_size:
            input_size = input_size if isinstance(input_size, tuple) else (input_size, input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for `PatchEmbed`.

        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

            - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
            - out_size (tuple[int]): Spatial shape of x, arrange as
              (out_h, out_w).
        """
        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class FFN(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        activation (Callable[..., nn.Module]): Activation layer module.
            Defaults to ``partial(nn.ReLU, inplace=True)``.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        ffn_drop: float = 0.0,
        dropout_layer: dict | None = None,
        add_identity: bool = True,
        init_cfg: dict | None = None,
    ):
        super().__init__(init_cfg)
        if num_fcs < 2:
            msg = "The number of fully-connected layers in FFNs should be at least 2."
            raise ValueError(msg)
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    nn.Linear(in_channels, feedforward_channels),
                    activation(),
                    nn.Dropout(ffn_drop),
                ),
            )
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)

        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        self.gamma2 = nn.Identity()

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


def deformable_attention_core_func(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor | list[tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Deformable attention core function.

    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = nn.functional.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(bs * n_head, 1, len_q, n_levels * n_points)
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, n_head * c, len_q)
    )

    return output.permute(0, 2, 1)
