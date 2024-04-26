# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utils for semantic segmentation."""

from __future__ import annotations

import warnings

import torch
from torch.nn import functional as f

from .blocks import OnnxLpNormalization


def resize(
    input_tensor: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | tuple[float, float] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
    warning: bool = True,
) -> torch.Tensor:
    """Resize the input tensor to the given size or according to a scale factor.

    Args:
        input_tensor (torch.Tensor): The input tensor to be resized.
        size (Optional[Tuple[int, int]]): The target size of the output tensor.
        scale_factor (Optional[Union[float, Tuple[float, float]]]): The scaling factor for
            the dimensions of the output tensor.
        mode (str): The interpolation mode. Default is "nearest".
        align_corners (Optional[bool]): Whether to align corners
            when mode is "linear" or "bilinear". Default is None.
        warning (bool): Whether to show a warning when align_corners is True
            and the output size is not a multiple of the input size. Default is True.

    Returns:
        torch.Tensor: The resized input tensor.

    """
    if warning and size is not None and align_corners:
        input_h, input_w = tuple(int(x) for x in input_tensor.shape[2:])
        output_h, output_w = tuple(int(x) for x in size)
        if (output_h > input_h or output_w > output_h) and (
            (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
            and (output_h - 1) % (input_h - 1)
            and (output_w - 1) % (input_w - 1)
        ):
            warnings.warn(
                f"When align_corners={align_corners}, "
                "the output would more aligned if "
                f"input size {(input_h, input_w)} is `x+1` and "
                f"out size {(output_h, output_w)} is `nx+1`",
                stacklevel=1,
            )
    return f.interpolate(input_tensor, size, scale_factor, mode, align_corners)


def normalize(x: torch.Tensor, dim: int, p: int = 2, eps: float = 1e-12) -> torch.Tensor:
    """Normalize method."""
    if torch.onnx.is_in_onnx_export():
        return OnnxLpNormalization.apply(x, dim, p, eps)
    return torch.nn.functional.normalize(x, dim=dim, p=p, eps=eps)


def channel_shuffle(
    x: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Channel Shuffle operation.

    This function enables cross-group information flow for multiple groups
    convolution layers.

    Args:
        x (torch.Tensor): The input tensor of shape
            (batch_size, num_channels, height, width).
        groups (int): The number of groups to divide the input tensor
            in the channel dimension.

    Returns:
        torch.Tensor: The output tensor after channel shuffle operation.
    """
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    return x.view(batch_size, groups * channels_per_group, height, width)
