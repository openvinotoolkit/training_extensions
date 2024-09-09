# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
#
"""ROI criterion."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from otx.algo.common.losses import CrossSigmoidFocalLoss


class ROICriterion(nn.Module):
    """ROICriterion is a loss criterion used in the Region of Interest (ROI) algorithm.

    Args:
        num_classes (int): Number of classes.
        bbox_coder (nn.Module): Bounding box coder.
        loss_cls (nn.Module): Classification loss.
        loss_mask (nn.Module): Mask loss.
        loss_bbox (nn.Module): Bounding box loss.
        class_agnostic (bool): Whether to use class agnostic
    """

    def __init__(
        self,
        num_classes: int,
        bbox_coder: nn.Module,
        loss_cls: nn.Module,
        loss_mask: nn.Module,
        loss_bbox: nn.Module,
        class_agnostic: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.bbox_coder = bbox_coder
        self.loss_bbox = loss_bbox
        self.loss_cls = loss_cls
        self.loss_mask = loss_mask
        self.use_sigmoid_cls = loss_cls.use_sigmoid
        self.class_agnostic = class_agnostic

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
        bbox_weights: Tensor,
        mask_preds: Tensor,
        mask_targets: Tensor,
        pos_labels: Tensor,
        valid_label_mask: Tensor | None = None,
        reduction_override: str | None = None,
    ) -> dict[str, Tensor]:
        """Forward function for ROI loss.

        Args:
            cls_score (Tensor): Classification score.
            bbox_pred (Tensor): Bounding box prediction.
            labels (Tensor): Labels.
            label_weights (Tensor): Label weights.
            bbox_targets (Tensor): Bounding box targets.
            bbox_weights (Tensor): Bounding box weights.
            mask_preds (Tensor): Mask prediction.
            mask_targets (Tensor): Mask targets.
            pos_labels (Tensor): Positive labels.
            valid_label_mask (Tensor, optional): Valid label mask. Defaults to None.
            reduction_override (str, optional): Reduction method. Defaults to None.
        """
        losses = {}
        if cls_score is not None and cls_score.numel() > 0:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.0)

            if isinstance(self.loss_cls, CrossSigmoidFocalLoss):
                losses["loss_cls"] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override,
                    valid_label_mask=valid_label_mask,
                )
            else:
                losses["loss_cls"] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override,
                )

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.class_agnostic:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[
                        pos_inds.type(torch.bool),
                        labels[pos_inds.type(torch.bool)],
                    ]
                losses["loss_bbox"] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override,
                )
            else:
                losses["loss_bbox"] = bbox_pred[pos_inds].sum()

        if mask_preds is not None:
            if mask_preds.size(0) == 0:
                loss_mask = mask_preds.sum()
            elif self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)

            losses["loss_mask"] = loss_mask

        return losses
