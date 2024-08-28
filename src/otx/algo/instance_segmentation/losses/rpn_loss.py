# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""RPN criterion."""

from __future__ import annotations

from torch import Tensor, nn

from otx.algo.common.utils.utils import multi_apply


class RPNCriterion(nn.Module):
    """RPNCriterion is a loss criterion used in the Region Proposal Network (RPN) algorithm.

    Args:
        num_classes (int): The number of object classes.
        bbox_coder (nn.Module): The module used for encoding and decoding bounding box coordinates.
        loss_cls (nn.Module): The module used for calculating the classification loss.
        loss_bbox (nn.Module): The module used for calculating the bounding box regression loss.
        loss_centerness (nn.Module | None, optional): The module used for calculating the centerness loss.
            Defaults to None.
        use_qfl (bool, optional): Whether to use the Quality Focal Loss (QFL).
            Defaults to ``CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)``.
        reg_decoded_bbox (bool, optional): Whether to use the decoded bounding box coordinates
            for regression loss calculation. Defaults to True.
        bg_loss_weight (float, optional): The weight for the background loss.
            Defaults to -1.0.
    """

    def __init__(
        self,
        bbox_coder: nn.Module,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
    ) -> None:
        super().__init__()
        self.bbox_coder = bbox_coder
        self.loss_bbox = loss_bbox
        self.loss_cls = loss_cls
        self.cls_out_channels = 1 if loss_cls.use_sigmoid else 2

    def forward(
        self,
        cls_reg_targets: tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor], int],
        bbox_preds: list[Tensor],
        cls_scores: list[Tensor],
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[InstanceData]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[InstanceData], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, avg_factor) = cls_reg_targets

        losses_cls, losses_bbox = multi_apply(
            self._forward,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor,
        )
        return {"loss_cls_rpn": losses_cls, "loss_bbox_rpn": losses_bbox}

    def _forward(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        avg_factor: int,
    ) -> tuple:
        """Calculate the loss of a single scale level based on the features extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.bbox_coder.encode_size)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls, loss_bbox
