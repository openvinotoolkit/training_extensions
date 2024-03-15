"""Module for defining IB Loss which alleviate effect of imbalanced dataset."""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models.losses import CrossEntropyLoss


@LOSSES.register_module()
class IBLoss(CrossEntropyLoss):
    """IB Loss, Influence-Balanced Loss for Imbalanced Visual Classification, https://arxiv.org/abs/2110.02444."""

    def __init__(self, num_classes, start=5, alpha=1000.0, reduction: str = "mean"):
        """Init fuction of IBLoss.

        Args:
            num_classes (int): Number of classes in dataset
            start (int): Epoch to start finetuning with IB loss
            alpha (float): Hyper-parameter for an adjustment for IB loss re-weighting
            reduction (str): How to reduce the output. Available options are "none" or "mean". Defaults to 'mean'.
        """
        super().__init__(loss_weight=1.0, reduction=reduction)
        if alpha < 0:
            raise ValueError("Alpha for IB loss should be bigger than 0")
        self.alpha = alpha
        self.epsilon = 0.001
        self.num_classes = num_classes
        self.register_buffer("weight", torch.ones(size=(self.num_classes,)))
        self._start_epoch = start
        self._cur_epoch = 0
        if reduction not in {"mean", "none"}:
            raise ValueError(f"reduction={reduction} is not allowed.")

    @property
    def cur_epoch(self):
        """Return current epoch."""
        return self._cur_epoch

    @cur_epoch.setter
    def cur_epoch(self, epoch):
        self._cur_epoch = epoch

    def update_weight(self, cls_num_list):
        """Update loss weight per class."""
        if len(cls_num_list) == 0:
            raise ValueError("Cannot compute the IB loss weight with empty cls_num_list.")
        per_cls_weights = 1.0 / (np.array(cls_num_list) + self.epsilon)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights)
        self.weight.data = per_cls_weights.to(device=self.weight.device)

    def forward(self, x, target, feature):
        """Forward fuction of IBLoss."""
        if self._cur_epoch < self._start_epoch:
            return super().forward(x, target)
        grads = torch.sum(torch.abs(F.softmax(x, dim=1) - F.one_hot(target, self.num_classes)), 1)
        feature = torch.sum(torch.abs(feature), 1).reshape(-1, 1)
        scaler = grads * feature.reshape(-1)
        scaler = self.alpha / (scaler + self.epsilon)
        ce_loss = F.cross_entropy(x, target, weight=self.weight, reduction="none")
        loss = ce_loss * scaler
        return loss.mean() if self.reduction == "mean" else loss
