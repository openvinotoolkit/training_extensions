# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models.losses import CrossEntropyLoss
from mmcls.models.losses.utils import weight_reduce_loss


def cross_entropy(pred, label, weight=None, reduction="mean", avg_factor=None, class_weight=None, ignore_index=None):
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
    def __init__(self, reduction="mean", loss_weight=1.0, ignore_index=None):
        super(CrossEntropyLossWithIgnore, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

        self.cls_criterion = cross_entropy

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
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", class_weight=None, loss_weight=1.0):
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

    def forward(self, cls_score, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            class_weight=self.class_weight,
            **kwargs
        )
        return loss_cls


class CrossEntropySmoothLoss(nn.Module):
    def __init__(self, epsilon=0.1, weight=1.0):
        super(CrossEntropySmoothLoss, self).__init__()
        self.epsilon = epsilon
        self.weight = weight

    def __call__(self, logits, target):
        with torch.no_grad():
            b, n, h, w = logits.size()
            assert n > 1

            target_value = 1.0 - self.epsilon
            blank_value = self.epsilon / float(n - 1)

            targets = logits.new_full((b * h * w, n), blank_value).scatter_(1, target.view(-1, 1), target_value)
            targets = targets.view(b, h, w, n).permute(0, 3, 1, 2)

        log_softmax = F.log_softmax(logits, dim=1)
        losses = torch.neg((targets * log_softmax).sum(dim=1))

        return self.weight * losses


class NormalizedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(NormalizedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, logits, target):
        log_softmax = F.log_softmax(logits, dim=1)
        b, c, h, w = log_softmax.size()

        log_softmax = log_softmax.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.view(-1)

        target_log_softmax = log_softmax[torch.arange(target.size(0), device=target.device), target]
        target_log_softmax = target_log_softmax.view(b, h, w)

        sum_log_softmax = log_softmax.sum(dim=1)
        losses = self.weight * target_log_softmax / sum_log_softmax

        return losses


class ReverseCrossEntropyLoss(nn.Module):
    def __init__(self, scale=4.0, weight=1.0):
        super(ReverseCrossEntropyLoss, self).__init__()
        self.weight = weight * abs(float(scale))

    def forward(self, logits, target):
        all_probs = F.softmax(logits, dim=1)
        b, c, h, w = all_probs.size()

        all_probs = all_probs.permute(0, 2, 3, 1).reshape(-1, c)
        target = target.view(-1)

        target_probs = all_probs[torch.arange(target.size(0), device=target.device), target]
        target_probs = target_probs.view(b, h, w)

        losses = self.weight * (1.0 - target_probs)

        return losses


class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.ce = CrossEntropyLoss(
            use_sigmoid=False,
            use_soft=False,
            reduction="none",
            loss_weight=alpha,
        )
        self.rce = ReverseCrossEntropyLoss(weight=beta)

    def forward(self, logits, target):
        return self.ce(logits, target) + self.rce(logits, target)


class ActivePassiveLoss(nn.Module):
    def __init__(self, alpha=100.0, beta=1.0):
        super(ActivePassiveLoss, self).__init__()
        self.active_loss = NormalizedCrossEntropyLoss(weight=alpha)
        self.passive_loss = ReverseCrossEntropyLoss(weight=beta)

    def forward(self, logits, target):
        return self.active_loss(logits, target) + self.passive_loss(logits, target)
