# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""YOLO v9 criterion."""

from __future__ import annotations
from typing import Any
import torch

from torch import nn, Tensor
from torch.nn import BCEWithLogitsLoss

from otx.algo.detection.utils.yolov7_v9_utils import Vec2Box, calculate_iou


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 assing pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Tensor:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def forward(
        self,
        predicts_bbox: Tensor,
        targets_bbox: Tensor,
        valid_masks: Tensor,
        box_norm: Tensor,
        cls_norm: Tensor,
    ) -> Tensor:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        return (loss_iou * box_norm).sum() / cls_norm


class DFLoss(nn.Module):
    def __init__(self, vec2box: Vec2Box, reg_max: int = 16) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Tensor:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0, self.reg_max - 1.01
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
    def __init__(
        self,
        class_num: int,
        anchors: Tensor,
        iou: str = "CIoU",
        topk: int = 10,
        factor: dict[str, float] = {"iou": 6.0, "cls": 0.5},
    ) -> None:
        self.class_num = class_num
        self.anchors = anchors
        self.iou = iou
        self.topk = topk
        self.factor = factor

    def get_valid_matrix(self, target_bbox: Tensor):
        """
        Get a boolean mask that indicates whether each target bounding box overlaps with each anchor.

        Args:
            target_bbox [batch x targets x 4]: The bounding box of each targets.
        Returns:
            [batch x targets x anchors]: A boolean tensor indicates if target bounding box overlaps with anchors.
        """
        Xmin, Ymin, Xmax, Ymax = target_bbox[:, :, None].unbind(3)
        anchors = self.anchors[None, None]  # add a axis at first, second dimension
        anchors_x, anchors_y = anchors.unbind(dim=3)
        target_in_x = (Xmin < anchors_x) & (anchors_x < Xmax)
        target_in_y = (Ymin < anchors_y) & (anchors_y < Ymax)
        target_on_anchor = target_in_x & target_in_y
        return target_on_anchor

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        """
        Get the (predicted class' probabilities) corresponding to the target classes across all anchors

        Args:
            predict_cls [batch x anchors x class]: The predicted probabilities for each class across each anchor.
            target_cls [batch x targets]: The class index for each target.

        Returns:
            [batch x targets x anchors]: The probabilities from `pred_cls` corresponding to the class indices specified in `target_cls`.
        """
        predict_cls = predict_cls.transpose(1, 2)
        target_cls = target_cls.expand(-1, -1, predict_cls.size(2))
        cls_probabilities = torch.gather(predict_cls, 1, target_cls)
        return cls_probabilities

    def get_iou_matrix(self, predict_bbox, target_bbox) -> Tensor:
        """
        Get the IoU between each target bounding box and each predicted bounding box.

        Args:
            predict_bbox [batch x predicts x 4]: Bounding box with [x1, y1, x2, y2].
            target_bbox [batch x targets x 4]: Bounding box with [x1, y1, x2, y2].
        Returns:
            [batch x targets x predicts]: The IoU scores between each target and predicted.
        """
        return calculate_iou(target_bbox, predict_bbox, self.iou).clamp(0, 1)

    def filter_topk(self, target_matrix: Tensor, topk: int = 10) -> tuple[Tensor, Tensor]:
        """
        Filter the top-k suitability of targets for each anchor.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors
            topk (int, optional): Number of top scores to retain per anchor.

        Returns:
            topk_targets [batch x targets x anchors]: Only leave the topk targets for each anchor
            topk_masks [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.
        """
        values, indices = target_matrix.topk(topk, dim=-1)
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_masks = topk_targets > 0
        return topk_targets, topk_masks

    def filter_duplicates(self, target_matrix: Tensor):
        """
        Filter the maximum suitability target index of each anchor.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors

        Returns:
            unique_indices [batch x anchors x 1]: The index of the best targets for each anchors
        """
        # TODO: add a assert for no target on the image
        unique_indices = target_matrix.argmax(dim=1)
        return unique_indices[..., None]

    def __call__(self, target: Tensor, predict: tuple[Tensor]) -> tuple[Tensor, Tensor]:
        """
        1. For each anchor prediction, find the highest suitability targets
        2. Select the targets
        2. Noramlize the class probilities of targets
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

        # delete one anchor pred assign to mutliple gts
        unique_indices = self.filter_duplicates(topk_targets)

        # TODO: do we need grid_mask? Filter the valid groud truth
        valid_mask = (grid_mask.sum(dim=-2) * topk_mask.sum(dim=-2)).bool()

        align_bbox = torch.gather(target_bbox, 1, unique_indices.repeat(1, 1, 4))
        align_cls = torch.gather(target_cls, 1, unique_indices).squeeze(-1)
        align_cls = nn.functional.one_hot(align_cls, self.class_num)

        # normalize class ditribution
        max_target = target_matrix.amax(dim=-1, keepdim=True)
        max_iou = iou_mat.amax(dim=-1, keepdim=True)
        normalize_term = (target_matrix / (max_target + 1e-9)) * max_iou
        normalize_term = normalize_term.permute(0, 2, 1).gather(2, unique_indices)
        align_cls = align_cls * normalize_term * valid_mask[:, :, None]

        return torch.cat([align_cls, align_bbox], dim=-1), valid_mask.bool()


class YOLOv9Criterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        vec2box: Vec2Box,
        loss_cls: nn.Module | None = None,
        loss_dfl: nn.Module | None = None,
        loss_iou: nn.Module | None = None,
        reg_max: int = 16,
        iou_rate: float = 0.5,
        dfl_rate: float = 7.5,
        cls_rate: float = 1.5,
        aux_rate: float = 0.25,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.loss_cls = loss_cls or BCELoss()
        self.loss_dfl = loss_dfl or DFLoss(vec2box, reg_max)
        self.loss_iou = loss_iou or BoxLoss()
        self.vec2box = vec2box
        self.matcher = BoxMatcher(num_classes, vec2box.anchor_grid)

        self.iou_rate = iou_rate
        self.dfl_rate = dfl_rate
        self.cls_rate = cls_rate
        self.aux_rate = aux_rate

    def forward(
        self,
        aux_predicts: list[Tensor],
        main_predicts: list[Tensor],
        targets: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        aux_predicts = self.vec2box(aux_predicts)
        main_predicts = self.vec2box(main_predicts)

        aux_iou, aux_dfl, aux_cls = self._forward(aux_predicts, targets)
        main_iou, main_dfl, main_cls = self._forward(main_predicts, targets)

        loss_dict = {
            "loss_cls": self.cls_rate * (aux_cls * self.aux_rate + main_cls),
            "loss_df": self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            "loss_iou": self.iou_rate * (aux_iou * self.aux_rate + main_iou),
        }
        loss_dict.update(loss=sum(list(loss_dict.values())) / len(loss_dict))
        return loss_dict

    def _forward(self, predicts: list[Tensor], targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.num_classes, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box
