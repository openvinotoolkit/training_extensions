# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn
import torch.nn.functional as F

from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None, class_weight=None, ignore_index=None):
    # element-wise losses
    if ignore_index is not None:
        loss = F.cross_entropy(pred, label, reduction='none', weight=class_weight, ignore_index=ignore_index)
    else:
        loss = F.cross_entropy(pred, label, reduction='none', weight=class_weight)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


@LOSSES.register_module()
class CrossEntropyLossWithIgnore(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0, ignore_index=None):
        super(CrossEntropyLossWithIgnore, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            ignore_index=self.ignore_index,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


@LOSSES.register_module()
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        super(WeightedCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cls_criterion = cross_entropy
        self.class_weight = class_weight
        if self.class_weight is not None:
            import torch
            self.class_weight = torch.tensor(self.class_weight)
            if torch.cuda.is_available():
                self.class_weight = self.class_weight.cuda()

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=self.class_weight,
            **kwargs)
        return loss_cls
