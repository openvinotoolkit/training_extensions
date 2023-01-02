# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models.losses.utils import weight_reduce_loss


def focal_loss(pred, target, weight=None, gamma=2.0, alpha=0.25, reduction="mean", avg_factor=None, ignore_index=None):
    if ignore_index is not None:
        pred = pred[target != ignore_index]
        target = target[target != ignore_index]
    pred_softmax = F.softmax(pred, 1)
    one_hot = torch.zeros_like(pred)
    one_hot = one_hot.scatter(1, target.view(-1, 1), 1)
    target = one_hot.type_as(pred)
    label_smoothing = 0.8
    target = target * label_smoothing + (1 - label_smoothing) / (target.shape[1])
    pt = (1 - pred_softmax) * target + pred_softmax * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def polarity_loss(
    pred, target, weight=None, gamma=2.0, alpha=0.25, beta=20, reduction="mean", avg_factor=None, ignore_index=None
):

    if ignore_index is not None:
        pred = pred[target != ignore_index]
        target = target[target != ignore_index]

    pred_softmax = F.softmax(pred, 1)
    one_hot = torch.zeros_like(pred)
    one_hot = one_hot.scatter(1, target.view(-1, 1), 1)
    target = one_hot.type_as(pred)
    label_smoothing = 1.0
    target = target * label_smoothing + (1 - label_smoothing) / (target.shape[1])
    penalty = F.sigmoid(beta * (pred_softmax - torch.sum(pred_softmax * target, dim=1, keepdim=True)))
    pt = (1 - pred_softmax) * target + pred_softmax * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight * penalty
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean", loss_weight=1.0, ignore_index=None):
        super(SoftmaxFocalLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

        self.cls_criterion = focal_loss

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
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


@LOSSES.register_module()
class SoftmaxPolarityLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, beta=20, reduction="mean", loss_weight=1.0, ignore_index=None):
        super(SoftmaxPolarityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index

        self.cls_criterion = polarity_loss

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
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
