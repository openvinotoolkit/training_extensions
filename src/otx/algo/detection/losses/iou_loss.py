# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.losses.iou_loss.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/losses/iou_loss.py
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor, nn

from otx.algo.common.losses.utils import weighted_loss
from otx.algo.common.utils.bbox_overlaps import bbox_overlaps


@weighted_loss
def iou_loss(pred: Tensor, target: Tensor, linear: bool = False, mode: str = "log", eps: float = 1e-6) -> Tensor:
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Epsilon to avoid log(0).

    Returns:
        Tensor: Loss tensor.
    """
    assert mode in ["linear", "square", "log"]  # noqa: S101
    if linear:
        mode = "linear"
        warnings.warn(
            "DeprecationWarning: Setting linear=True in iou_loss is deprecated, please use mode='linear' instead.",
            stacklevel=2,
        )
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)

    if fp16:
        ious = ious.to(torch.float16)

    if mode == "linear":
        loss = 1 - ious
    elif mode == "square":
        loss = 1 - ious**2
    elif mode == "log":
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(
        self,
        linear: bool = False,
        eps: float = 1e-6,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        mode: str = "log",
    ) -> None:
        super().__init__()
        assert mode in ["linear", "square", "log"]  # noqa: S101
        if linear:
            mode = "linear"
            warnings.warn(
                "DeprecationWarning: Setting linear=True in IOULoss is deprecated, please use mode='linear' instead.",
                stacklevel=2,
            )
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        assert reduction_override in (None, "none", "mean", "sum")  # noqa: S101
        reduction = reduction_override if reduction_override else self.reduction
        if (weight is not None) and (not torch.any(weight > 0)) and (reduction != "none"):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO (mmdet): remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape  # noqa: S101
            weight = weight.mean(-1)
        return self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs,
        )
