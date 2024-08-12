# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utils for semantic segmentation."""

from __future__ import annotations

import warnings
import math

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


def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    """Sinkhorn distribution."""
    L = torch.exp(out / epsilon).t()  # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    L = f.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    r"""Truncated normal distribution.

    Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w).
    """
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def momentum_update(old_value, new_value, momentum, debug=False):
    """EMA update function."""
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print(
            "old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
                momentum,
                torch.norm(old_value, p=2),
                (1 - momentum),
                torch.norm(new_value, p=2),
                torch.norm(update, p=2),
            )
        )
    return update
