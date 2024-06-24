# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Cross Focal Loss for ignore labels."""

from __future__ import annotations

import torch
import torch.nn.functional
from torch import Tensor, nn
from torch.cuda.amp import custom_fwd

from .focal_loss import py_sigmoid_focal_loss


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
        inputs (Tensor): inputs Tensor (N * C).
        targets (Tensor): targets Tensor (N).
        weight (Tensor, optional): weight Tensor (N), consists of (binarized label schema * weight).
        alpha (float): focal loss alpha.
        gamma (float): focal loss gamma.
        reduction (str): default = mean.
        avg_factor (float, optional): average factors.
        valid_label_mask (Tensor, optional): ignore label mask.
    """
    inputs_size = inputs.size(1)
    targets = torch.nn.functional.one_hot(targets, num_classes=inputs_size + 1)
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

    @custom_fwd(cast_inputs=torch.float32)
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
