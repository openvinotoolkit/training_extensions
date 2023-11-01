"""Cross Focal Loss for ignore labels."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import torch
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss, sigmoid_focal_loss
from mmdet.models.losses.varifocal_loss import varifocal_loss
from mmdet.registry import MODELS
from torch import nn
from torch.nn import functional

# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes, unused-argument


def cross_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    num_classes: int | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    avg_factor: int | torch.Tensor | None = None,
    use_vfl: bool = False,
    valid_label_mask: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Cross Focal Loss for ignore labels.

    Args:
        inputs: inputs Tensor (N * C).
        targets: targets Tensor (N), if use_vfl, then Tensor (N * C).
        weight: weight Tensor (N), consists of (binarized label schema * weight).
        num_classes: number of classes for training.
        alpha: focal loss alpha.
        gamma: focal loss gamma.
        reduction: default = mean.
        avg_factor: average factors.
        use_vfl: check use vfl.
        valid_label_mask: ignore label mask.
    """
    cross_mask = inputs.new_ones(inputs.shape, dtype=torch.int8)
    if valid_label_mask is not None:
        neg_mask = targets.sum(axis=1) == 0 if use_vfl else targets == num_classes
        neg_idx = neg_mask.nonzero(as_tuple=True)[0]
        cross_mask[neg_idx] = valid_label_mask[neg_idx].type(torch.int8)

    if use_vfl:
        calculate_loss_func = varifocal_loss
    elif torch.cuda.is_available() and inputs.is_cuda:
        calculate_loss_func = sigmoid_focal_loss
    else:
        inputs_size = inputs.size(1)
        targets = functional.one_hot(targets, num_classes=inputs_size + 1)
        targets = targets[:, :inputs_size]
        calculate_loss_func = py_sigmoid_focal_loss

    loss = (
        calculate_loss_func(inputs, targets, weight=weight, gamma=gamma, alpha=alpha, reduction="none", avg_factor=None)
        * cross_mask
    )

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
        num_classes: int | None = None,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
        ignore_index: int | None = None,
    ) -> None:
        """Initialize method.

        Args:
            use_sigmoid (bool): Whether use class-wise sigmoid or softmax
            num_classes (int | None): Number of classes for training.
            gamma (float): Focal loss gamma.
            alpha (float): Focal loss alpha.
            reduction (str): default = mean.
            loss_weight (float): Weight for focal loss
            ignore_index (int | None): Index for ignored class
        """
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid

        self.cls_criterion = cross_sigmoid_focal_loss

    def forward(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: int | None = None,
        use_vfl: bool = False,
        valid_label_mask: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward funtion of CrossSigmoidFocalLoss.

        Args:
            pred (torch.Tensor): Prediction results
            targets (torch.Tensor): Ground truth
            weight (torch.Tensor): Weight for loss
            reduction_override (str | None): Override for reduction
            avg_factor (int | None): Value for average the loss
            use_vfl (bool): Whether use vfl
            valid_label_mask (list[torch.Tensor]): Mask for valid labels
        """
        reduction = reduction_override if reduction_override else self.reduction
        return self.loss_weight * self.cls_criterion(
            pred,
            targets,
            weight=weight,
            num_classes=self.num_classes,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=reduction,
            avg_factor=avg_factor,
            use_vfl=use_vfl,
            valid_label_mask=valid_label_mask,
        )


@MODELS.register_module()
class OrdinaryFocalLoss(nn.Module):
    """Focal loss without balancing."""

    def __init__(self, gamma: float = 1.5) -> None:
        """Initialize method.

        Args:
            gamma (float): Focal loss gamma
        """
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        label_weights: torch.Tensor | None = None,
        avg_factor: int | torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Forward function for focal loss."""
        if target.numel() == 0:
            return 0.0 * pred.sum()

        ce = functional.cross_entropy(pred, target, reduction="none")
        p = torch.exp(-ce)
        loss = (1 - p) ** self.gamma * ce
        if label_weights is not None:
            loss = loss * label_weights
        if avg_factor is None:
            avg_factor = target.shape[0]
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.sum() / avg_factor
        return loss
