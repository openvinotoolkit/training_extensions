# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""RTMDet criterion."""

from __future__ import annotations

from torch import Tensor, nn

from otx.algo.common.utils.utils import multi_apply, reduce_mean


class RTMDetCriterion(nn.Module):
    """RTMDetCriterion is a criterion module for RTM-based object detection.

    Args:
        num_classes (int): Number of object classes.
        loss_cls (nn.Module): Classification loss module.
        loss_bbox (nn.Module): Bounding box regression loss module.
    """

    def __init__(self, num_classes: int, loss_cls: nn.Module, loss_bbox: nn.Module) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.use_sigmoid_cls = loss_cls.use_sigmoid
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            msg = f"num_classes={num_classes} is too small"
            raise ValueError(msg)

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
            cls_score (Tensor): Box scores for scale levels have shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for scale levels with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of anchors with shape (N, num_total_anchors).
            label_weights (Tensor): Label weights of anchors with shape (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of anchors with shape (N, num_total_anchors, 4).
            assign_metrics (Tensor): Assign metrics with shape (N, num_total_anchors).
            stride (list[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = multi_apply(
            self._forward,
            cls_score,
            bbox_pred,
            labels,
            label_weights,
            bbox_targets,
            assign_metrics,
            stride,
        )

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = [x / cls_avg_factor for x in losses_cls]

        bbox_avg_factor = reduce_mean(sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = [x / bbox_avg_factor for x in losses_bbox]
        return {"loss_cls": losses_cls, "loss_bbox": losses_bbox}

    def _forward(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        assign_metrics: Tensor,
        stride: list[int],
    ) -> tuple[Tensor, ...]:
        """Compute loss of a single scale level."""
        if stride[0] != stride[1]:
            msg = "h stride is not equal to w stride!"
            raise ValueError(msg)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        assign_metrics = assign_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)
        targets = (labels, assign_metrics)

        loss_cls = self.loss_cls(cls_score, targets, label_weights, avg_factor=1.0)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets

            # regression loss
            pos_bbox_weight = assign_metrics[pos_inds]

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0,
            )
        else:
            loss_bbox = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.0)

        return loss_cls, loss_bbox, assign_metrics.sum(), pos_bbox_weight.sum()
