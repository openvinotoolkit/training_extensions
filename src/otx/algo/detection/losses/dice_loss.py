"""Dice loss."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch
import torch.nn as nn

from otx.algo.detection.losses.weighted_loss import weight_reduce_loss


def dice_loss(
    pred,
    target,
    weight=None,
    eps=1e-3,
    reduction='mean',
    naive_dice=False,
    avg_factor=None,
):
    """Calculate dice loss, there are two forms of dice loss is supported.

    the one proposed in `V-Net: Fully Convolutional Neural
        Networks for Volumetric Medical Image Segmentation
        <https://arxiv.org/abs/1606.04797>`_.
    the dice loss in which the power of the number in the
        denominator is the first power instead of the second
        power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
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

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


class DiceLoss(nn.Module):
    """Dice loss."""
    def __init__(
        self,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=False,
        loss_weight=1.0,
        eps=1e-3,
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
        pred,
        target,
        weight=None,
        reduction_override=None,
        avg_factor=None,
    ) -> torch.Tensor:
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        if reduction_override not in (None, 'none', 'mean', 'sum'):
            msg = "reduction_override must be one of 'none', 'mean', 'sum'"
            raise ValueError(msg)
        reduction = (
            reduction_override if reduction_override else self.reduction)

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
            avg_factor=avg_factor)
