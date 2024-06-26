# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.losses.utils.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/losses/utils.py
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = functional._Reduction.get_enum(reduction)  # noqa: SLF001
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    if reduction_enum == 1:
        return loss.mean()
    if reduction_enum == 2:
        return loss.sum()
    msg = f"reduction_enum: {reduction_enum} is invalid"
    raise ValueError(msg)


def weight_reduce_loss(
    loss: Tensor,
    weight: Tensor | None = None,
    reduction: str = "mean",
    avg_factor: float | None = None,
) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor | None): Element-wise weights.
            Defaults to None.
        reduction (str): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (float | None): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then average the loss by avg_factor
    elif reduction == "mean":
        # Avoid causing ZeroDivisionError when avg_factor is 0.0,
        # i.e., all labels of an image belong to ignore index.
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)
    # if reduction is 'none', then do nothing, otherwise raise an error
    elif reduction != "none":
        msg = "avg_factor can not be used with reduction='sum'"
        raise ValueError(msg)
    return loss


def weighted_loss(loss_func: Callable) -> Callable:
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction: str = "mean",
        avg_factor: int | None = None,
        **kwargs,
    ) -> Tensor:
        """Wrapper for weighted loss.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): Target bboxes.
            weight (Tensor | None): The weight of loss for each
                prediction. Defaults to None.
            reduction (str): Options are "none", "mean" and "sum".
                Defaults to 'mean'.
            avg_factor (int | None): Average factor that is used
                to average the loss. Defaults to None.

        Returns:
            Tensor: Loss tensor.
        """
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        return weight_reduce_loss(loss, weight, reduction, avg_factor)

    return wrapper
