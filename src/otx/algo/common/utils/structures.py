# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Data structures for detection task.

Implementation modified from mmdet.models.task_modules.assigners.assign_result
and mmdet.models.task_modules.samplers.sampling_result.

Reference :
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/assigners/assign_result.py
    - https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/samplers/sampling_result.py
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class AssignResult:
    """Stores assignments between predicted and truth boxes.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/assigners/assign_result.py#L8-L198

    Args:
        num_gts (int): the number of truth boxes considered when computing this
            assignment
        gt_inds (Tensor): for each predicted box indicates the 1-based
            index of the assigned truth box. 0 means unassigned and -1 means
            ignore.
        max_overlaps (Tensor): the iou between the predicted box and its
            assigned truth box.
        labels (Tensor): If specified, for each predicted box
            indicates the category label of the assigned truth box.
    """

    def __init__(self, num_gts: int, gt_inds: Tensor, max_overlaps: Tensor, labels: Tensor) -> None:
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        # Interface for possible user-defined properties
        self._extra_properties: dict[str, Any] = {}

    @property
    def num_preds(self) -> int:
        """int: the number of predictions in this assignment."""
        return len(self.gt_inds)

    def set_extra_property(self, key: str, value: Any) -> None:  # noqa: ANN401
        """Set user-defined new property."""
        self._extra_properties[key] = value

    def get_extra_property(self, key: str) -> Any:  # noqa: ANN401
        """Get user-defined property."""
        return self._extra_properties.get(key, None)

    @property
    def info(self) -> dict:
        """Return a dictionary of info about the object."""
        basic_info = {
            "num_gts": self.num_gts,
            "num_preds": self.num_preds,
            "gt_inds": self.gt_inds,
            "max_overlaps": self.max_overlaps,
            "labels": self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def add_gt_(self, gt_labels: Tensor) -> None:
        """Add ground truth as assigned results.

        Args:
            gt_labels (Tensor): Labels of gt boxes
        """
        self_inds = torch.arange(1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat([self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps])

        self.labels = torch.cat([gt_labels, self.labels])


class SamplingResult:
    """Bbox sampling result.

    Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/samplers/sampling_result.py#L51-L179

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
        box_dim = 4
        if gt_bboxes.numel() == 0:
            self.pos_gt_bboxes = gt_bboxes.view(-1, box_dim)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, box_dim)
            self.pos_gt_bboxes = gt_bboxes[self.pos_assigned_gt_inds.long()]

    @property
    def priors(self) -> Tensor:
        """Tensor: concatenated positive and negative priors."""
        return torch.cat([self.pos_priors, self.neg_priors])

    @property
    def bboxes(self) -> Tensor:
        """Tensor: concatenated positive and negative boxes."""
        return self.priors

    @property
    def pos_bboxes(self) -> Tensor:
        """Return positive box pairs."""
        return self.pos_priors

    @property
    def neg_bboxes(self) -> Tensor:
        """Return negative box pairs."""
        return self.neg_priors

    def to(self, device: str | torch.device) -> SamplingResult:
        """Change the device of the data inplace.

        Example:
            >>> self = SamplingResult.random()
            >>> print(f'self = {self.to(None)}')
            >>> # xdoctest: +REQUIRES(--gpu)
            >>> print(f'self = {self.to(0)}')
        """
        _dict = self.__dict__
        for key, value in _dict.items():
            if isinstance(value, Tensor):
                _dict[key] = value.to(device)
        return self

    @property
    def info(self) -> dict:
        """Returns a dictionary of info about the object."""
        return {
            "pos_inds": self.pos_inds,
            "neg_inds": self.neg_inds,
            "pos_priors": self.pos_priors,
            "neg_priors": self.neg_priors,
            "pos_is_gt": self.pos_is_gt,
            "num_gts": self.num_gts,
            "pos_assigned_gt_inds": self.pos_assigned_gt_inds,
            "num_pos": self.num_pos,
            "num_neg": self.num_neg,
            "avg_factor": self.avg_factor,
        }
