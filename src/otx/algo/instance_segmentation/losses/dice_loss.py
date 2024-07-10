# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.losses.dice_loss.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/losses/dice_loss.py
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from otx.algo.common.losses.utils import weight_reduce_loss


def dice_loss(
    pred: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    eps: float = 1e-3,
    reduction: str = "mean",
    naive_dice: bool = False,
    avg_factor: int | None = None,
) -> Tensor:
    """Calculate dice loss, there are two forms of dice loss is supported.

    the one proposed in `V-Net: Fully Convolutional Neural
        Networks for Volumetric Medical Image Segmentation
        <https://arxiv.org/abs/1606.04797>`_.
    the dice loss in which the power of the number in the
        denominator is the first power instead of the second
        power.

    Args:
        pred (Tensor): The prediction, has a shape (n, *)
        target (Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(pred_input * target, 1)
    if naive_dice:
        b = torch.sum(pred_input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(pred_input * pred_input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        if weight.ndim != loss.ndim:
            msg = "weight must have the same number of dimensions as loss"
            raise ValueError(msg)
        if len(weight) != len(pred):
            msg = "The length of weight is not equal to the length of pred"
            raise ValueError(msg)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


class DiceLoss(nn.Module):
    """Dice loss."""

    def __init__(
        self,
        use_sigmoid: bool = True,
        activate: bool = True,
        reduction: str = "mean",
        naive_dice: bool = False,
        loss_weight: float = 1.0,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction_override: str | None = None,
        avg_factor: int | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction, has a shape (n, *).
            target (Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            Tensor: The calculated loss
        """
        if reduction_override not in (None, "none", "mean", "sum"):
            msg = "reduction_override must be one of 'none', 'mean', 'sum'"
            raise ValueError(msg)
        reduction = reduction_override if reduction_override else self.reduction

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        return self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor,
        )
