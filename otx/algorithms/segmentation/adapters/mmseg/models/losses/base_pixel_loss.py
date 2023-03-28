"""Base pixel loss."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from abc import abstractmethod

import torch
import torch.nn.functional as F
from mmseg.models.losses.utils import weight_reduce_loss

from otx.algorithms.segmentation.adapters.mmseg.utils.builder import (
    build_scalar_scheduler,
)

from .base_weighted_loss import BaseWeightedLoss


def entropy(p, dim=1, keepdim=False):
    """Calculates the entropy."""
    return -torch.where(p > 0.0, p * p.log(), torch.zeros_like(p)).sum(dim=dim, keepdim=keepdim)


class BasePixelLoss(BaseWeightedLoss):
    """Base pixel loss."""

    def __init__(self, scale_cfg=None, pr_product=False, conf_penalty_weight=None, border_reweighting=False, **kwargs):
        super().__init__(**kwargs)

        self._enable_pr_product = pr_product
        self._border_reweighting = border_reweighting

        self._reg_weight_scheduler = build_scalar_scheduler(conf_penalty_weight)
        self._scale_scheduler = build_scalar_scheduler(scale_cfg, default_value=1.0)

        self._last_scale = 0.0
        self._last_reg_weight = 0.0

    @property
    def last_scale(self):
        """Return last_scale."""
        return self._last_scale

    @property
    def last_reg_weight(self):
        """Return last_reg_weight."""
        return self._last_reg_weight

    @property
    def with_regularization(self):
        """Check regularization use."""
        return self._reg_weight_scheduler is not None

    @property
    def with_pr_product(self):
        """Check pr_product."""
        return self._enable_pr_product

    @property
    def with_border_reweighting(self):
        """Check border reweighting."""
        return self._border_reweighting

    @staticmethod
    def _pr_product(prod):
        alpha = torch.sqrt(1.0 - prod.pow(2.0))
        out_prod = alpha.detach() * prod + prod.detach() * (1.0 - alpha)

        return out_prod

    @staticmethod
    def _regularization(logits, scale, weight):
        probs = F.softmax(scale * logits, dim=1)
        entropy_values = entropy(probs, dim=1)
        out_values = -weight * entropy_values

        return out_values

    @staticmethod
    def _sparsity(values, valid_mask):
        with torch.no_grad():
            valid_values = values[valid_mask]
            sparsity = 1.0 - valid_values.count_nonzero() / max(1.0, valid_mask.sum())
        return sparsity.item()

    @staticmethod
    def _pred_stat(output, labels, valid_mask, window_size=5, min_group_ratio=0.6):
        assert window_size > 1
        assert 0.0 < min_group_ratio < 1.0

        min_group_size = int(min_group_ratio * window_size * window_size)
        assert min_group_size > 0

        with torch.no_grad():
            predictions = torch.argmax(output, dim=1)
            invalid_pred_mask = valid_mask & (predictions != labels)

            group_sizes = F.avg_pool2d(
                invalid_pred_mask.float(),
                kernel_size=window_size,
                stride=1,
                padding=(window_size - 1) // 2,
                divisor_override=1,
            )
            large_group_mask = invalid_pred_mask & (group_sizes >= min_group_size)

            num_target = torch.sum(large_group_mask, dim=(1, 2))
            num_total = torch.sum(invalid_pred_mask, dim=(1, 2))
            out_ratio = torch.mean(num_target / num_total.clamp_min(1))

        return out_ratio.item()

    def _forward(
        self, output, labels, avg_factor=None, pixel_weights=None, reduction_override=None
    ):  # pylint: disable=too-many-locals
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction

        self._last_scale = self._scale_scheduler(self.iter, self.epoch_size)

        if self.with_pr_product:
            output = self._pr_product(output)

        num_classes = output.size(1)
        valid_labels = torch.clamp(labels, 0, num_classes - 1)
        valid_mask = labels != self.ignore_index

        losses, updated_output = self._calculate(output, valid_labels, self._last_scale)

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
            weight = self.sampler(losses, output, valid_labels, valid_mask)
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

    @abstractmethod
    def _calculate(self, output, labels, scale):
        pass
