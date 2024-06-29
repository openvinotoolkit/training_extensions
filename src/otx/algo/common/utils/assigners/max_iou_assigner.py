# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.task_modules.assigners.max_iou_assigner.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/assigners/max_iou_assigner.py
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable

import torch
from otx.algo.common.utils.assigners import BboxOverlaps2D
from otx.algo.common.utils.structures import AssignResult
from torch import Tensor

if TYPE_CHECKING:
    from otx.algo.utils.mmengine_utils import InstanceData


class MaxIoUAssigner:
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule. More comparisons can be found in
            `PR #7464 <https://github.com/open-mmlab/mmdetection/pull/7464>`_.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
        iou_calculator (Callable): IoU calculator. Defaults to `BboxOverlaps2D()`.
        perm_repeat_gt_cfg (dict): Config of permute repeated gt bboxes.
    """

    def __init__(
        self,
        pos_iou_thr: float,
        neg_iou_thr: float | tuple,
        min_pos_iou: float = 0.0,
        gt_max_assign_all: bool = True,
        ignore_iof_thr: float = -1,
        ignore_wrt_candidates: bool = True,
        match_low_quality: bool = True,
        gpu_assign_thr: float = -1,
        iou_calculator: Callable | None = None,
        perm_repeat_gt_cfg: dict | None = None,
    ):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = iou_calculator or BboxOverlaps2D()
        self.perm_repeat_gt_cfg = perm_repeat_gt_cfg

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs,
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

        Args:
            pred_instances (InstanceData): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the `InstanceData`
                in other places.
            gt_instances (InstanceData): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (InstanceData, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            AssignResult: The assign result.
        """
        gt_bboxes = gt_instances.bboxes  # type: ignore[attr-defined]
        priors = pred_instances.priors  # type: ignore[attr-defined]
        gt_labels = gt_instances.labels  # type: ignore[attr-defined]
        gt_bboxes_ignore = gt_instances_ignore.bboxes if gt_instances_ignore is not None else None  # type: ignore[attr-defined]

        assign_on_cpu = (self.gpu_assign_thr > 0) and (gt_bboxes.shape[0] > self.gpu_assign_thr)
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()

        if self.perm_repeat_gt_cfg is not None and priors.numel() > 0:
            gt_bboxes_unique = perm_repeat_bboxes(gt_bboxes, self.perm_repeat_gt_cfg, self.iou_calculator)
        else:
            gt_bboxes_unique = gt_bboxes
        overlaps = self.iou_calculator(gt_bboxes_unique, priors)

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

    def assign_wrt_overlaps(self, overlaps: Tensor, gt_labels: Tensor) -> AssignResult:
        """Assign w.r.t. the overlaps of priors with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor): Labels of k gt_bboxes, shape (k, ).

        Returns:
            AssignResult: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            assigned_labels = overlaps.new_full((num_bboxes,), -1, dtype=torch.long)
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=max_overlaps,
                labels=assigned_labels,
            )

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0) & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0]) & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox 2.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=max_overlaps,
            labels=assigned_labels,
        )


def _perm_box(
    bboxes: Tensor,
    iou_calculator: Callable,
    iou_thr: float = 0.97,
    perm_range: float = 0.01,
    counter: int = 0,
    max_iter: int = 5,
) -> Tensor:
    """Compute the permuted bboxes.

    Args:
        bboxes (Tensor): Shape (n, 4) for , "xyxy" format.
        iou_calculator (obj): Overlaps Calculator.
        iou_thr (float): The permuted bboxes should have IoU > iou_thr.
        perm_range (float): The scale of permutation.
        counter (int): Counter of permutation iteration.
        max_iter (int): The max iterations of permutation.

    Returns:
        Tensor: The permuted bboxes.
    """
    ori_bboxes = copy.deepcopy(bboxes)
    is_valid = True
    batch_size = bboxes.size(0)
    perm_factor = bboxes.new_empty(batch_size, 4).uniform_(1 - perm_range, 1 + perm_range)
    bboxes *= perm_factor
    new_wh = bboxes[:, 2:] - bboxes[:, :2]
    if (new_wh <= 0).any():
        is_valid = False
    iou = iou_calculator(ori_bboxes.unique(dim=0), bboxes)
    if (iou < iou_thr).any():
        is_valid = False
    if not is_valid and counter < max_iter:
        return _perm_box(
            ori_bboxes,
            iou_calculator,
            perm_range=max(perm_range - counter * 0.001, 1e-3),
            counter=counter + 1,
        )
    return bboxes


def perm_repeat_bboxes(
    bboxes: Tensor,
    perm_repeat_cfg: dict,
    iou_calculator: Callable | None = None,
) -> Tensor:
    """Permute the repeated bboxes.

    Args:
        bboxes (Tensor): Shape (n, 4) for , "xyxy" format.
        iou_calculator (obj): Overlaps Calculator.
        perm_repeat_cfg (Dict | None): Config of permutation.

    Returns:
        Tensor: Bboxes after permuted repeated bboxes.
    """
    if iou_calculator is None:
        import torchvision

        iou_calculator = torchvision.ops.box_iou
    bboxes = copy.deepcopy(bboxes)
    unique_bboxes = bboxes.unique(dim=0)
    iou_thr = perm_repeat_cfg.get("iou_thr", 0.97)
    perm_range = perm_repeat_cfg.get("perm_range", 0.01)
    for box in unique_bboxes:
        inds = (bboxes == box).sum(-1).float() == 4
        if inds.float().sum().item() == 1:
            continue
        bboxes[inds] = _perm_box(bboxes[inds], iou_calculator, iou_thr=iou_thr, perm_range=perm_range, counter=0)
    return bboxes
