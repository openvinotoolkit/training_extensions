# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""ATSS criterion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from otx.algo.common.utils.utils import multi_apply
from otx.algo.detection.utils.utils import images_to_levels


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
        num_classes: int,
        bbox_coder: nn.Module,
        loss_cls: nn.Module,
        loss_bbox: nn.Module,
        reg_decoded_bbox: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bbox_coder = bbox_coder
        self.loss_bbox = loss_bbox
        self.loss_cls = loss_cls
        self.use_sigmoid_cls = loss_cls.use_sigmoid
        self.reg_decoded_bbox = reg_decoded_bbox

        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        if self.cls_out_channels <= 0:
            msg = f"num_classes={num_classes} is too small"
            raise ValueError(msg)

    def forward(
        self,
        cls_scores: list[Tensor],
        bbox_preds: list[Tensor],
        cls_reg_targets: list[Tensor],
        anchor_list: list[Tensor],
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection head.

        TODO (kirill): it is related to RPNHead for iseg, will be deprecated

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

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = [torch.cat(anchor) for anchor in anchor_list]
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self._forward,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor,
        )
        return {"loss_cls": losses_cls, "loss_bbox": losses_bbox}

    def _forward(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        anchors: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        avg_factor: int,
    ) -> tuple:
        """Calculate the loss of a single scale level based on the features extracted by the detection head.

        TODO (kirill): it is related to RPNHead for iseg, will be deprecated

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
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
        loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor)  # type: ignore[misc] # TODO (kirill): fix
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
        loss_bbox = self.loss_bbox(bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor)  # type: ignore[misc] # TODO (kirill): fix
        return loss_cls, loss_bbox
