"""Custom assigner for mmdet MaxIouAssigner."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mmdet.models.task_modules.assigners import AssignResult, MaxIoUAssigner
from mmdet.models.task_modules.builder import BBOX_ASSIGNERS
from torch import Tensor

if TYPE_CHECKING:
    from mmengine.structures import InstanceData


@BBOX_ASSIGNERS.register_module()
class CustomMaxIoUAssigner(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    This CustomMaxIoUAssigner patches assign funtion of mmdet's MaxIouAssigner
    so that it can prevent CPU OOM for images whose gt is extremely large
    """

    cpu_assign_thr = 1000

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
    ) -> AssignResult:
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        This CustomMaxIoUAssigner patches assign funtion of mmdet's MaxIouAssigner
        so that it can prevent CPU OOM for images whose gt is extremely large

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> from mmengine.structures import InstanceData
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> pred_instances = InstanceData()
            >>> pred_instances.priors = torch.Tensor([[0, 0, 10, 10],
            ...                                      [10, 10, 20, 20]])
            >>> gt_instances = InstanceData()
            >>> gt_instances.bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> gt_instances.labels = torch.Tensor([0])
            >>> assign_result = self.assign(pred_instances, gt_instances)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        gt_bboxes = gt_instances.bboxes
        priors = pred_instances.priors
        gt_labels = gt_instances.labels
        gt_bboxes_ignore = gt_instances_ignore.bboxes if gt_instances_ignore is not None else None

        assign_on_cpu = (self.gpu_assign_thr > 0) and (gt_bboxes.shape[0] > self.gpu_assign_thr)
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()

        if assign_on_cpu and gt_bboxes.shape[0] > self.cpu_assign_thr:
            split_length = gt_bboxes.shape[0] // self.cpu_assign_thr + 1
            overlaps: Tensor | list = []
            for i in range(split_length):
                gt_bboxes_split = gt_bboxes[i * self.cpu_assign_thr : (i + 1) * self.cpu_assign_thr]
                overlaps.append(self.iou_calculator(gt_bboxes_split, priors))
            overlaps = torch.concat(overlaps, dim=0)
        else:
            overlaps = self.iou_calculator(gt_bboxes, priors)

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and priors.numel() > 0
        ):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(priors, gt_bboxes_ignore, mode="iof")
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(gt_bboxes_ignore, priors, mode="iof")
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result
