"""Custom assigner for mmdet MaxIouAssigner."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdet.core.bbox.assigners import MaxIoUAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS


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

    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        Especially CustomMaxIoUAssigner split gt_bboxes tensor into small tensors
        when gt_bboxes is too large.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> bboxes = torch.Tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
            >>> gt_bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> assign_result = self.assign(bboxes, gt_bboxes)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        if assign_on_cpu and gt_bboxes.shape[0] > self.cpu_assign_thr:
            split_length = gt_bboxes.shape[0] // self.cpu_assign_thr + 1
            overlaps = []
            for i in range(split_length):
                gt_bboxes_split = gt_bboxes[i * self.cpu_assign_thr : (i + 1) * self.cpu_assign_thr]
                overlaps.append(self.iou_calculator(gt_bboxes_split, bboxes))
            overlaps = torch.concat(overlaps, dim=0)
        else:
            overlaps = self.iou_calculator(gt_bboxes, bboxes)

        if (
            self.ignore_iof_thr > 0
            and gt_bboxes_ignore is not None
            and gt_bboxes_ignore.numel() > 0
            and bboxes.numel() > 0
        ):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(bboxes, gt_bboxes_ignore, mode="iof")
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(gt_bboxes_ignore, bboxes, mode="iof")
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result
