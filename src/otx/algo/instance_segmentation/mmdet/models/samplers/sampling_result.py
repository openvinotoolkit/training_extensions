# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""MMDet SamplingResult."""
from __future__ import annotations

import warnings

import torch
from torch import Tensor

from otx.algo.instance_segmentation.mmdet.models.assigners import AssignResult


class SamplingResult:
    """Bbox sampling result.

    Args:
        pos_inds (Tensor): Indices of positive samples.
        neg_inds (Tensor): Indices of negative samples.
        priors (Tensor): The priors can be anchors or points,
            or the bboxes predicted by the previous stage.
        gt_bboxes (Tensor): Ground truth of bboxes.
        assign_result (:obj:`AssignResult`): Assigning results.
        gt_flags (Tensor): The Ground truth flags.
        avg_factor_with_neg (bool):  If True, ``avg_factor`` equal to
            the number of total priors; Otherwise, it is the number of
            positive priors. Defaults to True.

    Example:
        >>> # xdoctest: +IGNORE_WANT
        >>> from mmdet.models.task_modules.samplers.sampling_result import *  # NOQA
        >>> self = SamplingResult.random(rng=10)
        >>> print(f'self = {self}')
        self = <SamplingResult({
            'neg_inds': tensor([1,  2,  3,  5,  6,  7,  8,
                                9, 10, 11, 12, 13]),
            'neg_priors': torch.Size([12, 4]),
            'num_gts': 1,
            'num_neg': 12,
            'num_pos': 1,
            'avg_factor': 13,
            'pos_assigned_gt_inds': tensor([0]),
            'pos_inds': tensor([0]),
            'pos_is_gt': tensor([1], dtype=torch.uint8),
            'pos_priors': torch.Size([1, 4])
        })>
    """

    def __init__(
        self,
        pos_inds: Tensor,
        neg_inds: Tensor,
        priors: Tensor,
        gt_bboxes: Tensor,
        assign_result: AssignResult,
        gt_flags: Tensor,
        avg_factor_with_neg: bool = True,
    ) -> None:
        self.pos_inds = pos_inds
        self.neg_inds = neg_inds
        self.num_pos = max(pos_inds.numel(), 1)
        self.num_neg = max(neg_inds.numel(), 1)
        self.avg_factor_with_neg = avg_factor_with_neg
        self.avg_factor = self.num_pos + self.num_neg if avg_factor_with_neg else self.num_pos
        self.pos_priors = priors[pos_inds]
        self.neg_priors = priors[neg_inds]
        self.pos_is_gt = gt_flags[pos_inds]

        self.num_gts = gt_bboxes.shape[0]
        self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        self.pos_gt_labels = assign_result.labels[pos_inds]
        if gt_bboxes.numel() == 0:
            if self.pos_assigned_gt_inds.numel() != 0:
                msg = "gt_bboxes should not be empty"
                raise ValueError(msg)
            self.pos_gt_bboxes = gt_bboxes.view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long()]

    @property
    def priors(self) -> Tensor:
        """torch.Tensor: concatenated positive and negative priors."""
        return torch.cat([self.pos_priors, self.neg_priors], dim=0)

    @property
    def bboxes(self) -> Tensor:
        """torch.Tensor: concatenated positive and negative boxes."""
        warnings.warn("DeprecationWarning: bboxes is deprecated, please use 'priors' instead", stacklevel=2)
        return self.priors

    @property
    def pos_bboxes(self) -> Tensor:
        """Get positive bboxes."""
        warnings.warn("DeprecationWarning: pos_bboxes is deprecated, please use 'pos_priors' instead", stacklevel=2)
        return self.pos_priors

    @property
    def neg_bboxes(self) -> Tensor:
        """Get negative bboxes."""
        warnings.warn("DeprecationWarning: neg_bboxes is deprecated, please use 'neg_priors' instead", stacklevel=2)
        return self.neg_priors

    def to(self, device: torch.device | str) -> SamplingResult:
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, torch.Tensor):
                _dict[key] = value.to(device)
        return self
