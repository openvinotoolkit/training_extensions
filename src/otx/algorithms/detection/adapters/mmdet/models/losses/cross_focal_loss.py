"""Cross Focal Loss for ignore labels."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn.functional as F
from mmdet.models import LOSSES
from mmdet.models.losses.focal_loss import py_sigmoid_focal_loss, sigmoid_focal_loss
from mmdet.models.losses.varifocal_loss import varifocal_loss
from torch import nn

# pylint: disable=too-many-arguments, too-many-locals, too-many-instance-attributes, unused-argument


def cross_sigmoid_focal_loss(
    inputs,
    targets,
    weight=None,
    num_classes=None,
    alpha=0.25,
    gamma=2,
    reduction="mean",
    avg_factor=None,
    use_vfl=False,
    valid_label_mask=None,
):
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
        targets = F.one_hot(targets, num_classes=inputs_size + 1)
        targets = targets[:, :inputs_size]
        calculate_loss_func = py_sigmoid_focal_loss

    loss = (
        calculate_loss_func(inputs, targets, weight=weight, gamma=gamma, alpha=alpha, reduction="none", avg_factor=None)
        * cross_mask
    )

    if reduction == "mean":
        if avg_factor is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / avg_factor
    elif reduction == "sum":
        loss = loss.sum()
    return loss


@LOSSES.register_module()
class CrossSigmoidFocalLoss(nn.Module):
    """CrossSigmoidFocalLoss class for ignore labels with sigmoid."""

    def __init__(
        self,
        use_sigmoid=True,
        num_classes=None,
        gamma=2.0,
        alpha=0.25,
        reduction="mean",
        loss_weight=1.0,
        ignore_index=None,
    ):
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
        pred,
        targets,
        weight=None,
        reduction_override=None,
        avg_factor=None,
        use_vfl=False,
        valid_label_mask=None,
        **kwargs
    ):
        """Forward funtion of CrossSigmoidFocalLoss."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * self.cls_criterion(
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
        return loss_cls


@LOSSES.register_module()
class OrdinaryFocalLoss(nn.Module):
    """Focal loss without balancing."""

    def __init__(self, gamma=1.5, **kwargs):
        super(OrdinaryFocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma

    def forward(self, input, target, label_weights=None, avg_factor=None, reduction="mean", **kwars):
        """Forward function for focal loss."""
        if target.numel() == 0:
            return 0.0 * input.sum()

        CE = F.cross_entropy(input, target, reduction="none")
        p = torch.exp(-CE)
        loss = (1 - p) ** self.gamma * CE
        if label_weights is not None:
            assert len(loss) == len(label_weights)
            loss = loss * label_weights
        if avg_factor is None:
            avg_factor = target.shape[0]
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.sum() / avg_factor
        return loss
