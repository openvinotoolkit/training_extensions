# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Criterion module for YOLOv7 and v9.

Reference : https://github.com/WongKinYiu/YOLO
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from otx.algo.detection.utils.utils import Vec2Box


def calculate_iou(bbox1: Tensor, bbox2: Tensor, metrics: Literal["iou", "diou", "ciou"] = "iou") -> Tensor:
    """Calculate the Intersection over Union (IoU) between two sets of bounding boxes.

    Args:
        bbox1 (Tensor): The first set of bounding boxes.
        bbox2 (Tensor): The second set of bounding boxes.
        metrics (Literal["iou", "diou", "ciou"], optional): The metrics to calculate. Defaults to "iou".

    Returns:
        Tensor: The IoU between the two sets of bounding boxes.
    """
    eps = 1e-9
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    # Expand dimensions if necessary
    if bbox1.ndim == 2 and bbox2.ndim == 2:
        bbox1 = bbox1.unsqueeze(1)  # (Ax4) -> (Ax1x4)
        bbox2 = bbox2.unsqueeze(0)  # (Bx4) -> (1xBx4)
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZxAx4) -> (BZxAx1x4)
        bbox2 = bbox2.unsqueeze(1)  # (BZxBx4) -> (BZx1xBx4)

    # Calculate intersection coordinates
    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(ymax_inter - ymin_inter, min=0)

    # Calculate area of each bbox
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + eps)
    if metrics == "iou":
        return iou.to(dtype)

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(bbox1[..., 0], bbox2[..., 0])
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + eps

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou.to(dtype)

    # Compute aspect ratio penalty term
    arctan = torch.atan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + eps)) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + eps),
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    alpha = v / (v - iou + 1 + eps)
    # Compute CIoU
    ciou = diou - alpha * v
    return ciou.to(dtype)


class BCELoss(nn.Module):
    """Binary Cross Entropy Loss.

    TODO (author): Refactor the device, should be assign by config
    TODO (author): origin v9 assign pos_weight == 1?
    TODO (sungchul): check if it can be replaced with otx.algo.common.losses.cross_entropy_loss.CrossEntropyLoss
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Tensor:
        """Calculate the BCE loss for the classification."""
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    """Box Loss.

    TODO (sungchul): check if it can be replaced with otx.algo.common.losses.iou_loss.IoULoss
    """

    def forward(
        self,
        predicts_bbox: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        """Calculate the IoU loss for the bounding box.

        Args:
            predicts_bbox (Tensor): The predicted bounding box.
            targets_bbox (Tensor): The target bounding box.
            valid_masks (Tensor): The mask for valid bounding box.
            box_norm (Tensor): The normalization factor for the bounding box.
            cls_norm (Tensor): The normalization factor for the class.

        Returns:
            Tensor: The IoU loss for the bounding box.
        """
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        return (loss_iou * box_norm).sum() / cls_norm


class DFLoss(nn.Module):
    """Distribution Focal Loss (DFL).

    Args:
        vec2box (Vec2Box): The Vec2Box object.
        reg_max (int, optional): Maximum number of anchor regions. Defaults to 16.
    """

    def __init__(self, vec2box: Vec2Box, reg_max: int = 16) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self,
        predicts_anc: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        """Calculate the DFLoss for the bounding box.

        Args:
            predicts_anc (Tensor): The predicted anchor.
            targets_bbox (Tensor): The target bounding box.
            valid_masks (Tensor): The mask for valid bounding box.
            box_norm (Tensor): The normalization factor for the bounding box.
            cls_norm (Tensor): The normalization factor for the class.

        Returns:
            Tensor: The DFLoss for the bounding box.
        """
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0,
            self.reg_max - 1.01,
        )
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = nn.functional.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = nn.functional.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        return (loss_dfl * box_norm).sum() / cls_norm


class BoxMatcher:
    """Box Matcher.

    Args:
        class_num (int): The number of classes.
        anchors (Tensor): The anchor tensor.
        iou (str, optional): The IoU method. Defaults to "CIoU".
        topk (int, optional): The number of top scores to retain per anchor. Defaults to 10.
        factor (dict[str, float] | None, optional): The factor for IoU and class. Defaults to {"iou": 6.0, "cls": 0.5}.
    """

    def __init__(
        self,
        class_num: int,
        anchors: Tensor,
        iou: Literal["iou", "diou", "ciou"] = "ciou",
        topk: int = 10,
        factor: dict[str, float] | None = None,
    ) -> None:
        self.class_num = class_num
        self.anchors = anchors
        self.iou = iou
        self.topk = topk
        self.factor = factor or {"iou": 6.0, "cls": 0.5}

    def get_valid_matrix(self, target_bbox: Tensor) -> Tensor:
        """Get a boolean mask that indicates whether each target bounding box overlaps with each anchor.

        Args:
            target_bbox (Tensor): The bounding box of each targets with (batch, targets, 4).

        Returns:
            Tensor: A boolean tensor indicates if target bounding box overlaps with anchors
                with (batch, targets, anchors).
        """
        xmin, ymin, xmax, ymax = target_bbox[:, :, None].unbind(3)
        anchors = self.anchors[None, None]  # add a axis at first, second dimension
        anchors_x, anchors_y = anchors.unbind(dim=3)
        target_in_x = (xmin < anchors_x) & (anchors_x < xmax)
        target_in_y = (ymin < anchors_y) & (anchors_y < ymax)
        return target_in_x & target_in_y

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        """Get the (predicted class' probabilities) corresponding to the target classes across all anchors.

        Args:
            predict_cls (Tensor): The predicted probabilities for each class across each anchor
                with (batch, anchors, class).
            target_cls (Tensor): The class index for each target with (batch, targets, 1).

        Returns:
            Tensor: The probabilities from `pred_cls` corresponding to the class indices
                specified in `target_cls` with (batch, targets, anchors).
        """
        predict_cls = predict_cls.transpose(1, 2)
        target_cls = target_cls.expand(-1, -1, predict_cls.size(2))
        return torch.gather(predict_cls, 1, target_cls)

    def get_iou_matrix(self, predict_bbox: Tensor, target_bbox: Tensor) -> Tensor:
        """Get the IoU between each target bounding box and each predicted bounding box.

        Args:
            predict_bbox (Tensor): Bounding box with [x1, y1, x2, y2] with (batch, predicts, 4).
            target_bbox (Tensor): Bounding box with [x1, y1, x2, y2] with (batch, targets, 4).

        Returns:
            Tensor: The IoU scores between each target and predicted with (batch, targets, predicts).
        """
        return calculate_iou(target_bbox, predict_bbox, self.iou).clamp(0, 1)

    def filter_topk(self, target_matrix: Tensor, topk: int = 10) -> tuple[Tensor, Tensor]:
        """Filter the top-k suitability of targets for each anchor.

        Args:
            target_matrix (Tensor): The suitability for each targets-anchors with (batch, targets, anchors).
            topk (int, optional): Number of top scores to retain per anchor.

        Returns:
            tuple[Tensor, Tensor]: The top-k suitability for each targets-anchors with (batch, targets, anchors)
                and a boolean mask indicating the top-k scores' positions with (batch, targets, anchors).
        """
        values, indices = target_matrix.topk(topk, dim=-1)
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_masks = topk_targets > 0
        return topk_targets, topk_masks

    def filter_duplicates(self, target_matrix: Tensor) -> Tensor:
        """Filter the maximum suitability target index of each anchor.

        Args:
            target_matrix (Tensor): The suitability for each targets-anchors with (batch, targets, anchors).

        Returns:
            unique_indices (Tensor): The index of the best targets for each anchors with (batch, anchors, 1).
        """
        # TODO (author): add a assert for no target on the image
        unique_indices = target_matrix.argmax(dim=1)
        return unique_indices[..., None]

    def __call__(self, target: Tensor, predict: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        """Assign the best suitable ground truth box for each predicted anchor.

        1. For each anchor prediction, find the highest suitability targets
        2. Select the targets
        3. Normalize the class probabilities of targets.

        Args:
            target (Tensor): The target tensor with class and bounding box with (batch, targets, (class + 4)).
            predict (tuple[Tensor, Tensor]): The predicted class and bounding box.

        Returns:
            tuple[Tensor, Tensor]: The aligned target tensor with (batch, targets, (class + 4)).
        """
        predict_cls, predict_bbox = predict
        target_cls, target_bbox = target.split([1, 4], dim=-1)  # B x N x (C B) -> B x N x C, B x N x B
        target_cls = target_cls.long().clamp(0)

        # get valid matrix (each gt appear in which anchor grid)
        grid_mask = self.get_valid_matrix(target_bbox)

        # get iou matrix (iou with each gt bbox and each predict anchor)
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox)

        # get cls matrix (cls prob with each gt class and each predict class)
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls)

        target_matrix = grid_mask * (iou_mat ** self.factor["iou"]) * (cls_mat ** self.factor["cls"])

        # choose topk
        topk_targets, topk_mask = self.filter_topk(target_matrix, topk=self.topk)

        # delete one anchor pred assign to multiple gts
        unique_indices = self.filter_duplicates(topk_targets)

        # TODO (author): do we need grid_mask? Filter the valid ground truth
        valid_mask = (grid_mask.sum(dim=-2) * topk_mask.sum(dim=-2)).bool()

        align_bbox = torch.gather(target_bbox, 1, unique_indices.repeat(1, 1, 4))
        align_cls = torch.gather(target_cls, 1, unique_indices).squeeze(-1)
        align_cls = nn.functional.one_hot(align_cls, self.class_num)

        # normalize class distribution
        max_target = target_matrix.amax(dim=-1, keepdim=True)
        max_iou = iou_mat.amax(dim=-1, keepdim=True)
        normalize_term = (target_matrix / (max_target + 1e-9)) * max_iou
        normalize_term = normalize_term.permute(0, 2, 1).gather(2, unique_indices)
        align_cls = align_cls * normalize_term * valid_mask[:, :, None]

        return torch.cat([align_cls, align_bbox], dim=-1), valid_mask.bool()


class YOLOv9Criterion(nn.Module):
    """YOLOv9 criterion module.

    This module calculates the loss for YOLOv9 object detection model.

    Args:
        num_classes (int): The number of classes.
        vec2box (Vec2Box): The Vec2Box object.
        loss_cls (nn.Module | None): The classification loss module. Defaults to None.
        loss_dfl (nn.Module | None): The DFLoss loss module. Defaults to None.
        loss_iou (nn.Module | None): The IoULoss loss module. Defaults to None.
        reg_max (int, optional): Maximum number of anchor regions. Defaults to 16.
        cls_rate (float, optional): The classification loss rate. Defaults to 1.5.
        dfl_rate (float, optional): The DFLoss loss rate. Defaults to 7.5.
        iou_rate (float, optional): The IoU loss rate. Defaults to 0.5.
        aux_rate (float, optional): The auxiliary loss rate. Defaults to 0.25.
    """

    def __init__(
        self,
        num_classes: int,
        vec2box: Vec2Box,
        loss_cls: nn.Module | None = None,
        loss_dfl: nn.Module | None = None,
        loss_iou: nn.Module | None = None,
        reg_max: int = 16,
        cls_rate: float = 0.5,
        dfl_rate: float = 1.5,
        iou_rate: float = 7.5,
        aux_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls = loss_cls or BCELoss()
        self.loss_dfl = loss_dfl or DFLoss(vec2box, reg_max)
        self.loss_iou = loss_iou or BoxLoss()
        self.vec2box = vec2box
        self.matcher = BoxMatcher(num_classes, vec2box.anchor_grid)

        self.cls_rate = cls_rate
        self.dfl_rate = dfl_rate
        self.iou_rate = iou_rate
        self.aux_rate = aux_rate

    def forward(
        self,
        main_preds: tuple[Tensor, Tensor, Tensor],
        targets: Tensor,
        aux_preds: tuple[Tensor, Tensor, Tensor] | None = None,
    ) -> dict[str, Tensor] | None:
        """Forward pass of the YOLOv9 criterion module.

        Args:
            main_preds (tuple[Tensor, Tensor, Tensor]): The main predictions.
            targets (Tensor): The learning target of the prediction.
            aux_preds (tuple[Tensor, Tensor, Tensor], optional): The auxiliary predictions. Defaults to None.

        Returns:
            dict[str, Tensor]: The loss dictionary.
        """
        if targets.shape[1] == 0:
            # TODO (sungchul): should this step be done here?
            return None

        main_preds = self.vec2box(main_preds)
        main_iou, main_dfl, main_cls = self._forward(main_preds, targets)

        aux_iou, aux_dfl, aux_cls = 0.0, 0.0, 0.0
        if aux_preds:
            aux_preds = self.vec2box(aux_preds)
            aux_iou, aux_dfl, aux_cls = self._forward(aux_preds, targets)

        loss_dict = {
            "loss_cls": self.cls_rate * (aux_cls * self.aux_rate + main_cls),
            "loss_df": self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            "loss_iou": self.iou_rate * (aux_iou * self.aux_rate + main_iou),
        }
        loss_dict.update(total_loss=sum(list(loss_dict.values())) / len(loss_dict))
        return loss_dict

    def _forward(self, predicts: tuple[Tensor, Tensor, Tensor], targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()))

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = targets_cls.sum()
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.loss_cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.loss_iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.loss_dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        return loss_iou, loss_dfl, loss_cls

    def separate_anchor(self, anchors: Tensor) -> tuple[Tensor, Tensor]:
        """Separate anchor and bounding box.

        Args:
            anchors (Tensor): The anchor tensor.
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.num_classes, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box
