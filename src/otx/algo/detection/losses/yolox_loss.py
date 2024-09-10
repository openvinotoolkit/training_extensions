# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""YOLOX criterion."""

from __future__ import annotations

from torch import Tensor, nn

from otx.algo.common.losses import CrossEntropyLoss, IoULoss, L1Loss


class YOLOXCriterion(nn.Module):
    """YOLOX criterion module.

    This module calculates the loss for YOLOX object detection model.

    Args:
        num_classes (int): The number of classes.
        loss_cls (nn.Module | None): The classification loss module. Defaults to None.
        loss_bbox (nn.Module | None): The bounding box regression loss module. Defaults to None.
        loss_obj (nn.Module | None): The objectness loss module. Defaults to None.
        loss_l1 (nn.Module | None): The L1 loss module. Defaults to None.

    Returns:
        dict[str, Tensor]: A dictionary containing the calculated losses.

    """

    def __init__(
        self,
        num_classes: int,
        loss_cls: nn.Module | None = None,
        loss_bbox: nn.Module | None = None,
        loss_obj: nn.Module | None = None,
        loss_l1: nn.Module | None = None,
        use_l1: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls = loss_cls or CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0)
        self.loss_bbox = loss_bbox or IoULoss(mode="square", eps=1e-16, reduction="sum", loss_weight=5.0)
        self.loss_obj = loss_obj or CrossEntropyLoss(use_sigmoid=True, reduction="sum", loss_weight=1.0)
        self.loss_l1 = loss_l1 or L1Loss(reduction="sum", loss_weight=1.0)
        self.use_l1 = use_l1

    def forward(
        self,
        flatten_objectness: Tensor,
        flatten_cls_preds: Tensor,
        flatten_bbox_preds: Tensor,
        flatten_bboxes: Tensor,
        obj_targets: Tensor,
        cls_targets: Tensor,
        bbox_targets: Tensor,
        l1_targets: Tensor,
        num_total_samples: Tensor,
        num_pos: Tensor,
        pos_masks: Tensor,
    ) -> dict[str, Tensor]:
        """Forward pass of the YOLOX criterion module.

        Args:
            flatten_objectness (Tensor): Flattened objectness predictions.
            flatten_cls_preds (Tensor): Flattened class predictions.
            flatten_bbox_preds (Tensor): Flattened bounding box predictions.
            flatten_bboxes (Tensor): Flattened ground truth bounding boxes.
            obj_targets (Tensor): Objectness targets.
            cls_targets (Tensor): Class targets.
            bbox_targets (Tensor): Bounding box targets.
            l1_targets (Tensor): L1 targets.
            num_total_samples (Tensor): Total number of samples.
            num_pos (Tensor): Number of positive samples.
            pos_masks (Tensor): Positive masks.

        Returns:
            dict[str, Tensor]: A dictionary containing the calculated losses.

        """
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1), obj_targets) / num_total_samples
        if num_pos > 0:
            loss_cls = (
                self.loss_cls(flatten_cls_preds.view(-1, self.num_classes)[pos_masks], cls_targets) / num_total_samples
            )
            loss_bbox = self.loss_bbox(flatten_bboxes.view(-1, 4)[pos_masks], bbox_targets) / num_total_samples
        else:
            # Avoid cls and reg branch not participating in the gradient
            # propagation when there is no ground-truth in the images.
            # For more details, please refer to
            # https://github.com/open-mmlab/mmdetection/issues/7298
            loss_cls = flatten_cls_preds.sum() * 0
            loss_bbox = flatten_bboxes.sum() * 0

        loss_dict = {"loss_cls": loss_cls, "loss_bbox": loss_bbox, "loss_obj": loss_obj}

        if self.use_l1:
            if num_pos > 0:
                loss_l1 = self.loss_l1(flatten_bbox_preds.view(-1, 4)[pos_masks], l1_targets) / num_total_samples
            else:
                # Avoid cls and reg branch not participating in the gradient
                # propagation when there is no ground-truth in the images.
                # For more details, please refer to
                # https://github.com/open-mmlab/mmdetection/issues/7298
                loss_l1 = flatten_bbox_preds.sum() * 0
            loss_dict.update(loss_l1=loss_l1)

        return loss_dict
