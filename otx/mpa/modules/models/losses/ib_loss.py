# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import torch
import torch.nn.functional as F
from mmcls.models.builder import LOSSES
from mmcls.models.losses import CrossEntropyLoss


@LOSSES.register_module()
class IBLoss(CrossEntropyLoss):
    def __init__(self, num_classes, start=5, alpha=1000.0, **kwargs):
        """IB Loss
        https://arxiv.org/abs/2110.02444

        Args:
            num_classes (int): Number of classes in dataset
            start (int): Epoch to start finetuning with IB loss
            alpha (float): Hyper-parameter for an adjustment for IB loss re-weighting
        """
        super(IBLoss, self).__init__(loss_weight=1.0)
        if alpha < 0:
            raise ValueError("Alpha for IB loss should be bigger than 0")
        self.alpha = alpha
        self.epsilon = 0.001
        self.num_classes = num_classes
        self.weight = None
        self._start_epoch = start
        self._cur_epoch = 0

    @property
    def cur_epoch(self):
        return self._cur_epoch

    @cur_epoch.setter
    def cur_epoch(self, epoch):
        self._cur_epoch = epoch

    def update_weight(self, cls_num_list):
        if len(cls_num_list) == 0:
            raise ValueError("Cannot compute the IB loss weight with empty cls_num_list.")
        per_cls_weights = 1.0 / np.array(cls_num_list)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights)
        self.weight = per_cls_weights

    def forward(self, input, target, feature):
        if self._cur_epoch < self._start_epoch:
            return super().forward(input, target)
        else:
            grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, self.num_classes)), 1)
            feature = torch.sum(torch.abs(feature), 1).reshape(-1, 1)
            ib = grads * feature.reshape(-1)
            ib = self.alpha / (ib + self.epsilon)
            ce_loss = F.cross_entropy(input, target, weight=self.weight.to(input.get_device()), reduction="none")
            loss = ce_loss * ib
            return loss.mean()
