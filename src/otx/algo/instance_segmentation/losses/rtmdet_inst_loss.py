# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""RTMDet for instance segmentation criterion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.detection.losses import RTMDetCriterion

if TYPE_CHECKING:
    from torch import Tensor, nn


class RTMDetInstCriterion(RTMDetCriterion):
    """Criterion of RTMDet for instance segmentation.

    Args:
        num_classes (int): Number of object classes.
        loss_cls (nn.Module): Classification loss module.
        loss_bbox (nn.Module): Bounding box regression loss module.
        loss_mask (nn.Module): Mask loss module.
    """

    def __init__(
        self,
        num_classes: int,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        loss_mask: nn.Module,
    ) -> None:
        super().__init__(num_classes, loss_cls, loss_bbox)
        self.loss_mask = loss_mask

    def forward(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        assign_metrics: Tensor,
        stride: list[int],
        **kwargs,
    ) -> dict[str, Tensor]:
        """Compute loss of a single scale level.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape
                (N, num_total_anchors).
            stride (list[int]): Downsample stride of the feature map.
            batch_pos_mask_logits (Tensor): The prediction, has a shape (n, *).
            pos_gt_masks (Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            num_pos (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        loss_dict = super().forward(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            assign_metrics=assign_metrics,
            stride=stride,
        )

        if (num_pos := kwargs.pop("num_pos")) == 0:
            zero_loss: Tensor = kwargs.pop("zero_loss")
            loss_dict.update({"loss_mask": zero_loss})
            return loss_dict

        batch_pos_mask_logits: Tensor = kwargs.pop("batch_pos_mask_logits")
        pos_gt_masks: Tensor = kwargs.pop("pos_gt_masks")
        loss_mask = self.loss_mask(batch_pos_mask_logits, pos_gt_masks, weight=None, avg_factor=num_pos)
        loss_dict.update({"loss_mask": loss_mask})
        return loss_dict
