# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""ATSS criterion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from otx.algo.common.losses import CrossEntropyLoss, CrossSigmoidFocalLoss, QualityFocalLoss
from otx.algo.common.utils.bbox_overlaps import bbox_overlaps
from otx.algo.common.utils.utils import multi_apply, reduce_mean


class ATSSCriterion(nn.Module):
    """ATSSCriterion is a loss criterion used in the Adaptive Training Sample Selection (ATSS) algorithm.

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
        loss_centerness: nn.Module | None = None,
        use_qfl: bool = False,
        qfl_cfg: dict | None = None,
        reg_decoded_bbox: bool = True,
        bg_loss_weight: float = -1.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bbox_coder = bbox_coder
        self.use_qfl = use_qfl
        self.reg_decoded_bbox = reg_decoded_bbox
        self.bg_loss_weight = bg_loss_weight

        self.loss_bbox = loss_bbox
        self.loss_centerness = loss_centerness or CrossEntropyLoss(use_sigmoid=True, loss_weight=1.0)

        if use_qfl:
            loss_cls = qfl_cfg or QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0)

        self.loss_cls = loss_cls

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
        anchors: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        centerness: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        valid_label_mask: Tensor,
        avg_factor: float,
    ) -> dict[str, Tensor]:
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for scale levels with shape (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for scale levels have shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for scale levels with shape (N, num_anchors * 4, H, W).
            centerness(Tensor): Centerness scores for each scale level.
            labels (Tensor): Labels of anchors with shape (N, num_total_anchors).
            label_weights (Tensor): Label weights of anchors with shape (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of anchors with shape (N, num_total_anchors, 4).
            valid_label_mask (Tensor): Label mask for consideration of ignored label
                with shape (N, num_total_anchors, 1).
            avg_factor (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        losses_cls, losses_bbox, loss_centerness, bbox_avg_factor = multi_apply(
            self._forward,
            anchors,
            cls_score,
            bbox_pred,
            centerness,
            labels,
            label_weights,
            bbox_targets,
            valid_label_mask,
            avg_factor=avg_factor,
        )

        bbox_avg_factor = sum(bbox_avg_factor)
        bbox_avg_factor = reduce_mean(bbox_avg_factor).clamp_(min=1).item()
        losses_bbox = [loss_bbox / bbox_avg_factor for loss_bbox in losses_bbox]
        return {"loss_cls": losses_cls, "loss_bbox": losses_bbox, "loss_centerness": loss_centerness}

    def _forward(
        self,
        anchors: Tensor,
        cls_score: Tensor,
        bbox_pred: Tensor,
        centerness: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        valid_label_mask: Tensor,
        avg_factor: float,
    ) -> tuple:
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            centerness(Tensor): Centerness scores for each scale level.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            valid_label_mask (Tensor): Label mask for consideration of ignored
                label with shape (N, num_total_anchors, 1).
            avg_factor (float): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            tuple[Tensor]: A tuple of loss components.
        """
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        centerness = centerness.permute(0, 2, 3, 1).reshape(-1)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        valid_label_mask = valid_label_mask.reshape(-1, self.cls_out_channels)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = self._get_pos_inds(labels)

        if self.use_qfl:
            quality = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_centerness = centerness[pos_inds]

            centerness_targets = self.centerness_target(pos_anchors, pos_bbox_targets)
            if self.reg_decoded_bbox:
                pos_bbox_pred = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)

            if self.use_qfl:
                quality[pos_inds] = bbox_overlaps(pos_bbox_pred.detach(), pos_bbox_targets, is_aligned=True).clamp(
                    min=1e-6,
                )

            # regression loss
            loss_bbox = self._get_loss_bbox(pos_bbox_targets, pos_bbox_pred, centerness_targets)

            # centerness loss
            loss_centerness = self._get_loss_centerness(avg_factor, pos_centerness, centerness_targets)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_centerness = centerness.sum() * 0
            centerness_targets = bbox_targets.new_tensor(0.0)

        # Re-weigting BG loss
        if self.bg_loss_weight >= 0.0:
            neg_indices = labels == self.num_classes
            label_weights[neg_indices] = self.bg_loss_weight

        if self.use_qfl:
            labels = (labels, quality)  # For quality focal loss arg spec

        # classification loss
        loss_cls = self._get_loss_cls(cls_score, labels, label_weights, valid_label_mask, avg_factor)

        return loss_cls, loss_bbox, loss_centerness, centerness_targets.sum()

    def centerness_target(self, anchors: Tensor, gts: Tensor) -> Tensor:
        """Calculate the centerness between anchors and gts.

        Only calculate pos centerness targets, otherwise there may be nan.

        Args:
            anchors (Tensor): Anchors with shape (N, 4), "xyxy" format.
            gts (Tensor): Ground truth bboxes with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Centerness between anchors and gts.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        l_ = anchors_cx - gts[:, 0]
        t_ = anchors_cy - gts[:, 1]
        r_ = gts[:, 2] - anchors_cx
        b_ = gts[:, 3] - anchors_cy

        left_right = torch.stack([l_, r_], dim=1)
        top_bottom = torch.stack([t_, b_], dim=1)
        return torch.sqrt(
            (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
            * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0]),
        )

    def _get_pos_inds(self, labels: Tensor) -> Tensor:
        bg_class_ind = self.num_classes
        return ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

    def _get_loss_bbox(
        self,
        pos_bbox_targets: Tensor,
        pos_bbox_pred: Tensor,
        centerness_targets: Tensor,
    ) -> Tensor:
        return self.loss_bbox(pos_bbox_pred, pos_bbox_targets, weight=centerness_targets, avg_factor=1.0)

    def _get_loss_centerness(
        self,
        avg_factor: Tensor,
        pos_centerness: Tensor,
        centerness_targets: Tensor,
    ) -> Tensor:
        return self.loss_centerness(pos_centerness, centerness_targets, avg_factor=avg_factor)

    def _get_loss_cls(
        self,
        cls_score: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        valid_label_mask: Tensor,
        avg_factor: Tensor,
    ) -> Tensor:
        if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
            loss_cls = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                valid_label_mask=valid_label_mask,
            )
        else:
            loss_cls = self.loss_cls(cls_score, labels, label_weights, avg_factor=avg_factor)
        return loss_cls
