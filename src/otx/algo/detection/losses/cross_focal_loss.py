# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Cross Focal Loss for ignore labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss, sigmoid_focal_loss
from mmdet.registry import MODELS
from torch import Tensor, nn


def cross_sigmoid_focal_loss(
    inputs: Tensor,
    targets: Tensor,
    weight: Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    avg_factor: float | None = None,
    valid_label_mask: Tensor | None = None,
) -> Tensor:
    """Cross Focal Loss for ignore labels.

    Args:
        inputs: inputs Tensor (N * C).
        targets: targets Tensor (N).
        weight: weight Tensor (N), consists of (binarized label schema * weight).
        alpha: focal loss alpha.
        gamma: focal loss gamma.
        reduction: default = mean.
        avg_factor: average factors.
        valid_label_mask: ignore label mask.
    """
    if torch.cuda.is_available() and inputs.is_cuda:
        calculate_loss_func = sigmoid_focal_loss
    else:
        inputs_size = inputs.size(1)
        targets = F.one_hot(targets, num_classes=inputs_size + 1)
        targets = targets[:, :inputs_size]
        calculate_loss_func = py_sigmoid_focal_loss

    loss = calculate_loss_func(
        inputs,
        targets,
        weight=weight,
        gamma=gamma,
        alpha=alpha,
        reduction="none",
        avg_factor=None,
    )

    loss = loss * valid_label_mask if valid_label_mask is not None else loss

    if reduction == "mean":
        loss = loss.mean() if avg_factor is None else loss.sum() / avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@MODELS.register_module()
class CrossSigmoidFocalLoss(nn.Module):
    """CrossSigmoidFocalLoss class for ignore labels with sigmoid."""

    def __init__(
        self,
        use_sigmoid: bool = True,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.use_sigmoid = use_sigmoid

        self.cls_criterion = cross_sigmoid_focal_loss

    def forward(
        self,
        pred: Tensor,
        targets: Tensor,
        weight: Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: float | None = None,
        valid_label_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Forward funtion of CrossSigmoidFocalLoss."""
        if reduction_override not in (None, "none", "mean", "sum"):
            msg = f"{reduction_override} is not in (None, none, mean, sum)"
            raise ValueError(msg)
        reduction = reduction_override if reduction_override else self.reduction
        return self.loss_weight * self.cls_criterion(
            pred,
            targets,
            weight=weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor,
            valid_label_mask=valid_label_mask,
        )


@MODELS.register_module()
class OrdinaryFocalLoss(nn.Module):
    """Focal loss without balancing."""

    def __init__(self, gamma: float = 1.5, **kwargs):
        super().__init__()
        if gamma < 0:
            msg = f"{gamma} is not valid number for gamma."
            raise ValueError(msg)
        self.gamma = gamma

    def forward(
        self,
        inputs: Tensor,
        targets: Tensor,
        label_weights: Tensor | None = None,
        avg_factor: float | None = None,
        reduction: str = "mean",
        **kwargs,
    ) -> Tensor:
        """Forward function for focal loss."""
        if targets.numel() == 0:
            return 0.0 * inputs.sum()

        cross_entropy_value = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-cross_entropy_value)
        loss = (1 - p) ** self.gamma * cross_entropy_value
        if label_weights is not None:
            loss = loss * label_weights
        if avg_factor is None:
            avg_factor = targets.shape[0]
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.sum() / avg_factor
        return loss
