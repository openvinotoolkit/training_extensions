# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmseg.models.losses.utils import weight_reduce_loss

from .base_pixel_loss import BasePixelLoss


class MPABasePixelLoss(BasePixelLoss):
    def __init__(self, **kwargs):
        super(MPABasePixelLoss, self).__init__(**kwargs)

    def _forward(
        self,
        output,
        labels,
        valid_label_mask,
        avg_factor=None,
        pixel_weights=None,
        reduction_override=None,
    ):
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        self._last_scale = self._scale_scheduler(self.iter, self.epoch_size)

        if self.with_pr_product:
            output = self._pr_product(output)

        import numpy as np

        _labels = labels.cpu().detach().numpy()
        _labels[np.where((_labels == self.ignore_index))] = 0
        num_classes = output.size(1)
        valid_labels = torch.clamp(labels, 0, num_classes - 1)
        valid_mask = labels != self.ignore_index

        losses, updated_output = self._calculate(output, _labels, valid_label_mask, self._last_scale)

        if self.with_regularization:
            self._last_reg_weight = self._reg_weight_scheduler(self.iter, self.epoch_size)
            regularization = self._regularization(updated_output, self._last_scale, self._last_reg_weight)
            losses = torch.clamp_min(losses + regularization, 0.0)

        if self.with_border_reweighting:
            assert pixel_weights is not None
            losses = pixel_weights.squeeze(1) * losses

        losses = torch.where(valid_mask, losses, torch.zeros_like(losses))
        raw_sparsity = self._sparsity(losses, valid_mask)
        invalid_ratio = self._pred_stat(output, labels, valid_mask)

        weight, weight_sparsity = None, 0.0
        if self.sampler is not None:
            weight = self.sampler.sample(output, valid_labels, losses, valid_mask)
            weight_sparsity = self._sparsity(weight, valid_mask)

        loss = weight_reduce_loss(losses, weight=weight, reduction=reduction, avg_factor=avg_factor)

        meta = dict(
            weight=self.last_loss_weight,
            reg_weight=self.last_reg_weight,
            scale=self.last_scale,
            raw_sparsity=raw_sparsity,
            weight_sparsity=weight_sparsity,
            invalid_ratio=invalid_ratio,
        )

        return loss, meta
