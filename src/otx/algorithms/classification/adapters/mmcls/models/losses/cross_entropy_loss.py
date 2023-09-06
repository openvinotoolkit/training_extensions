"""Module for defining cross entropy loss for classification task."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weight_reduce_loss
from torch import nn


def cross_entropy(pred, label, weight=None, reduction="mean", avg_factor=None, class_weight=None, ignore_index=None):
    """Calculate cross entropy for given pred, label pairs."""
    # element-wise losses
    if ignore_index is not None:
        loss = F.cross_entropy(pred, label, reduction="none", weight=class_weight, ignore_index=ignore_index)
    else:
        loss = F.cross_entropy(pred, label, reduction="none", weight=class_weight)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class CrossEntropyLossWithIgnore(nn.Module):
    """Defining CrossEntropyLossWothIgnore which supports ignored_label masking."""

    def __init__(self, reduction="mean", loss_weight=1.0, ignore_index=None):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        self.cls_criterion = cross_entropy

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        """Forward function of CrossEntropyLossWithIgnore class."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            ignore_index=self.ignore_index,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs
        )
        return loss_cls
