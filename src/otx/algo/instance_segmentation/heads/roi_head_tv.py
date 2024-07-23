# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Customised torchvision RoIHeads class with support for polygons as ground truth masks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as f
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, maskrcnn_inference

from otx.algo.instance_segmentation.utils.structures.mask import mask_target

if TYPE_CHECKING:
    from datumaro import Polygon


def maskrcnn_loss(
    mask_logits: Tensor,
    proposals: list[Tensor],
    gt_masks: list[list[Tensor]] | list[list[Polygon]],
    gt_labels: list[Tensor],
    mask_matched_idxs: list[Tensor],
    image_shapes: list[tuple[int, int]],
) -> Tensor:
    """Compute the mask prediction loss."""
    cfg = {"mask_size": mask_logits.shape[-1]}
    meta_infos = [{"img_shape": img_shape} for img_shape in image_shapes]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]

    mask_targets = mask_target(
        proposals,
        mask_matched_idxs,
        gt_masks,
        cfg,
        meta_infos,
    )

    labels = torch.cat(labels, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    return f.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],  # type: ignore[attr-defined]
        mask_targets,
    )


class TVRoIHeads(RoIHeads):
    """Customised RoIHeads class with support for polygons as ground truth masks."""

    def forward(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[list[dict[str, Tensor]], dict[str, Tensor]]:
        """Support both polygons and masks as ground truth masks.

        Note: This method is a copy of the original forward method from RoIHeads.
        TODO(Eugene): Add support for incremental learning.
        """
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            if labels is None:
                msg = "labels cannot be None"
                raise ValueError
            if regression_targets is None:
                msg = "regression_targets cannot be None"
                raise ValueError(msg)
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            result = [
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
                for i in range(num_images)
            ]

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is None:
                    msg = "if in training, matched_idxs should not be None"
                    raise ValueError(msg)

                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                msg = "Expected mask_roi_pool to be not None"
                raise RuntimeError(msg)

            loss_mask = {}
            if self.training:
                if targets is None or pos_matched_idxs is None or mask_logits is None:
                    msg = "targets, pos_matched_idxs, mask_logits cannot be None when training"
                    raise ValueError(msg)

                gt_masks = (
                    [t["masks"] for t in targets] if len(targets[0]["masks"]) else [t["polygons"] for t in targets]
                )
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits,
                    mask_proposals,
                    gt_masks,
                    gt_labels,
                    pos_matched_idxs,
                    image_shapes,
                )
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        return result, losses

    def export(
        self,
        features: dict[str, Tensor],
        proposals: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Export the model for inference."""
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)

        mask_features = self.mask_roi_pool(
            features,
            boxes,
            image_shapes,
        )
        mask_features = self.mask_head(mask_features)
        mask_logits = self.mask_predictor(mask_features)

        masks_probs = maskrcnn_inference(mask_logits, labels)
        masks_probs = [masks_probs[0].squeeze(1)]

        boxes = [torch.cat([boxes[0], scores[0].unsqueeze(-1)], -1)]
        return boxes, labels, masks_probs
