# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Dynamic Soft Label Assigner for RTMDet."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional
from torch import Tensor

from otx.algo.detection.heads.iou2d_calculator import BboxOverlaps2D
from otx.algo.detection.utils.structures import AssignResult

INF = 100000000
EPS = 1.0e-7
BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit

if TYPE_CHECKING:
    from otx.algo.utils.mmengine_utils import InstanceData


def center_of_mass(masks: Tensor, eps: float = 1e-7) -> Tensor:
    """Compute the masks center of mass.

    Args:
        masks: Mask tensor, has shape (num_masks, H, W).
        eps: a small number to avoid normalizer to be zero.
            Defaults to 1e-7.


    Returns:
        Tensor: The masks center of mass. Has shape (num_masks, 2).
    """
    n, h, w = masks.shape
    device = masks.device
    gpu_usage = np.ceil(n * int(h) * int(w) * BYTES_PER_FLOAT)
    num_chunks = int(gpu_usage / GPU_MEM_LIMIT) if gpu_usage > GPU_MEM_LIMIT else n

    if num_chunks > n:
        msg = "num_chunks should be less than n"
        raise ValueError(msg)

    indices = torch.arange(n, device=device)
    chunks = torch.chunk(indices, num_chunks) if num_chunks != n else [indices]
    grid_h = torch.arange(h, device=device)[:, None]
    grid_w = torch.arange(w, device=device)
    center = torch.zeros(n, 2, device=device, dtype=torch.float32)
    for inds in chunks:
        masks_chunk = masks[inds]
        normalizer = masks_chunk.sum(dim=(1, 2)).float().clamp(min=eps)
        center_y = (masks_chunk * grid_h).sum(dim=(1, 2)) / normalizer
        center_x = (masks_chunk * grid_w).sum(dim=(1, 2)) / normalizer
        center[inds] = torch.cat([center_x[:, None], center_y[:, None]], dim=1)
    return center


class DynamicSoftLabelAssigner:
    """Computes matching between predictions and ground truth with dynamic soft label assignment.

    Args:
        soft_center_radius (float): Radius of the soft center prior.
            Defaults to 3.0.
        topk (int): Select top-k predictions to calculate dynamic k
            best matches for each gt. Defaults to 13.
        iou_weight (float): The scale factor of iou cost. Defaults to 3.0.
        iou_calculator (ConfigType): Config of overlaps Calculator.
            Defaults to dict(type='BboxOverlaps2D').
    """

    def __init__(
        self,
        soft_center_radius: float = 3.0,
        topk: int = 13,
        iou_weight: float = 3.0,
    ) -> None:
        self.soft_center_radius = soft_center_radius
        self.topk = topk
        self.iou_weight = iou_weight
        self.iou_calculator = BboxOverlaps2D()

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: InstanceData | None = None,
        **kwargs,
    ) -> AssignResult:
        """Assign gt to priors.

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
            kwargs (dict, optional): Any keyword arguments to be used.


        Returns:
            obj:`AssignResult`: The assigned result.
        """
        gt_bboxes = gt_instances.bboxes  # type: ignore[attr-defined]
        gt_labels = gt_instances.labels  # type: ignore[attr-defined]
        num_gt = gt_bboxes.size(0)

        decoded_bboxes = pred_instances.bboxes  # type: ignore[attr-defined]
        pred_scores = pred_instances.scores  # type: ignore[attr-defined]
        priors = pred_instances.priors  # type: ignore[attr-defined]
        num_bboxes = decoded_bboxes.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        prior_center = priors[:, :2]

        # Tensor boxes will be treated as horizontal boxes by defaults
        lt_ = prior_center[:, None] - gt_bboxes[:, :2]
        rb_ = gt_bboxes[:, 2:] - prior_center[:, None]

        deltas = torch.cat([lt_, rb_], dim=-1)
        deltas_values, _ = deltas.min(dim=-1)
        is_in_gts = deltas_values > 0

        valid_mask = is_in_gts.sum(dim=1) > 0

        valid_decoded_bbox = decoded_bboxes[valid_mask]
        valid_pred_scores = pred_scores[valid_mask]
        num_valid = valid_decoded_bbox.size(0)

        if num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_bboxes.new_zeros((num_bboxes,))
            assigned_labels = decoded_bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
            return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        if hasattr(gt_instances, "masks"):
            gt_center = center_of_mass(gt_instances.masks, eps=EPS)
        else:
            # Tensor boxes will be treated as horizontal boxes by defaults
            gt_center = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2.0
        valid_prior = priors[valid_mask]
        strides = valid_prior[:, 2]
        distance = (valid_prior[:, None, :2] - gt_center[None, :, :]).pow(2).sum(-1).sqrt() / strides[:, None]
        soft_center_prior = torch.pow(10, distance - self.soft_center_radius)

        pairwise_ious = self.iou_calculator(valid_decoded_bbox, gt_bboxes)
        iou_cost = -torch.log(pairwise_ious + EPS) * self.iou_weight

        gt_onehot_label = (
            torch.nn.functional.one_hot(gt_labels.to(torch.int64), pred_scores.shape[-1])
            .float()
            .unsqueeze(0)
            .repeat(num_valid, 1, 1)
        )
        valid_pred_scores = valid_pred_scores.unsqueeze(1).repeat(1, num_gt, 1)

        soft_label = gt_onehot_label * pairwise_ious[..., None]
        scale_factor = soft_label - valid_pred_scores.sigmoid()
        soft_cls_cost = torch.nn.functional.binary_cross_entropy_with_logits(
            valid_pred_scores,
            soft_label,
            reduction="none",
        ) * scale_factor.abs().pow(2.0)
        soft_cls_cost = soft_cls_cost.sum(dim=-1)

        cost_matrix = soft_cls_cost + iou_cost + soft_center_prior

        matched_pred_ious, matched_gt_inds = self.dynamic_k_matching(cost_matrix, pairwise_ious, num_gt, valid_mask)

        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes,), -INF, dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious
        return AssignResult(num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def dynamic_k_matching(
        self,
        cost: Tensor,
        pairwise_ious: Tensor,
        num_gt: int,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Use IoU and matching cost to calculate the dynamic top-k positive targets. Same as SimOTA.

        Args:
            cost (Tensor): Cost matrix.
            pairwise_ious (Tensor): Pairwise iou matrix.
            num_gt (int): Number of gt.
            valid_mask (Tensor): Mask for valid bboxes.

        Returns:
            tuple: matched ious and gt indexes.
        """
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix * pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
