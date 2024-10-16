# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""MaskDINO Instance Segmentation model.

Implementation modified from:
    * https://github.com/IDEA-Research/MaskDINO
    * https://github.com/facebookresearch/Mask2Former
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.ops.roi_align import RoIAlign

from otx.algo.instance_segmentation.heads import MaskDINODecoderHeadModule, MaskDINOEncoderHeadModule
from otx.core.data.entity.base import ImageInfo

if TYPE_CHECKING:
    from torchvision import tv_tensors
    from torchvision.models.detection.image_list import ImageList


class MaskDINOHead(nn.Module):
    """MaskDINO Head module.

    Args:
        num_classes (int): number of classes
        pixel_decoder (MaskDINOEncoderHeadModule): pixel decoder
        predictor (MaskDINODecoderHeadModule): mask transformer predictor
        num_queries (int): number of queries
        test_topk_per_image (int): number of topk per image
    """

    def __init__(
        self,
        num_classes: int,
        pixel_decoder: MaskDINOEncoderHeadModule,
        predictor: MaskDINODecoderHeadModule,
        num_queries: int = 300,
        test_topk_per_image: int = 100,
    ):
        super().__init__()
        self.pixel_decoder = pixel_decoder
        self.predictor = predictor
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.test_topk_per_image = test_topk_per_image

    def forward(
        self,
        features: dict[str, Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Forward pass."""
        mask_features, _, multi_scale_features = self.pixel_decoder(features)
        return self.predictor(multi_scale_features, mask_features, targets=targets)

    def predict(
        self,
        features: dict[str, Tensor],
        imgs_info: list[ImageInfo],
    ) -> dict[str, list[Tensor]]:
        """Predict."""
        outputs, _ = self(features)

        class_queries_logits = outputs["pred_logits"]
        masks_queries_logits = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]

        device = masks_queries_logits.device
        num_classes = self.num_classes
        num_queries = self.num_queries
        test_topk_per_image = self.test_topk_per_image

        batch_scores: list[Tensor] = []
        batch_bboxes: list[tv_tensors.BoundingBoxes] = []
        batch_labels: list[torch.LongTensor] = []
        batch_masks: list[tv_tensors.Mask] = []

        for mask_pred, mask_cls, pred_boxes, img_info in zip(
            masks_queries_logits,
            class_queries_logits,
            mask_box_results,
            imgs_info,
        ):
            ori_h, ori_w = img_info.ori_shape
            scores = mask_cls.sigmoid()
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes

            mask_pred = mask_pred[topk_indices]  # noqa: PLW2901
            pred_boxes = pred_boxes[topk_indices]  # noqa: PLW2901
            pred_scores = scores_per_image * self.calculate_mask_scores(mask_pred)
            pred_classes = labels_per_image

            pred_masks = (
                (
                    torch.nn.functional.interpolate(
                        mask_pred.unsqueeze(0),
                        size=(ori_h, ori_w),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                )
                > 0
            )

            pred_boxes = pred_boxes.new_tensor([[ori_w, ori_h, ori_w, ori_h]]) * box_convert(  # noqa: PLW2901
                pred_boxes,
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )
            pred_boxes[:, 0::2].clamp_(min=0, max=ori_w - 1)
            pred_boxes[:, 1::2].clamp_(min=0, max=ori_h - 1)

            area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
            keep = (pred_masks.sum((1, 2)) > 5) & (area > 10) & (pred_scores > 0.05)

            batch_masks.append(pred_masks[keep])
            batch_bboxes.append(pred_boxes[keep])
            batch_labels.append(pred_classes[keep])
            batch_scores.append(pred_scores[keep])

        return {
            "pred_boxes": batch_bboxes,
            "pred_labels": batch_labels,
            "pred_masks": batch_masks,
            "pred_scores": batch_scores,
        }

    def calculate_mask_scores(self, mask_pred: Tensor) -> Tensor:
        """Calculate mask scores."""
        pred_masks = (mask_pred > 0).to(mask_pred)

        # Calculate average mask prob
        return (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)


class MaskDINO(nn.Module):
    """Main class for mask classification semantic segmentation architectures."""

    def __init__(
        self,
        backbone: nn.Module,
        sem_seg_head: MaskDINOHead,
        criterion: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.roi_align = RoIAlign(
            output_size=(28, 28),
            sampling_ratio=0,
            aligned=True,
            spatial_scale=1.0,
        )

    def forward(
        self,
        images: ImageList,
        imgs_info: list[ImageInfo],
        targets: list[dict[str, Any]] | None = None,
        mode: str = "tensor",
    ) -> dict[str, Tensor] | dict[str, list[Tensor]]:
        """Forward pass."""
        features = self.backbone(images.tensors)

        if self.training:
            outputs, mask_dict = self.sem_seg_head(features, targets=targets)
            losses = self.criterion(outputs, targets, mask_dict)
            for k in list(losses.keys()):
                losses[k] *= self.criterion.weight_dict[k]
            return losses

        return self.sem_seg_head.predict(features, imgs_info)

    def roi_mask_extraction(
        self,
        bboxes: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """Extract masks from RoI (Region of Interest).

        This function is used for exporting the model, as it extracts same-size square masks from RoI for speed.

        Args:
            bboxes (Tensor): Bounding boxes with shape (N, 4), where N is the number of bounding boxes.
            masks (Tensor): Masks with shape (H, W), where H and W are the height and width of the mask.

        Returns:
            Tensor: Extracted masks with shape (1, N, H', W'), where H' and W'
                are the height and width of the extracted masks.
        """
        bboxes = bboxes.unsqueeze(0)
        batch_index = torch.arange(bboxes.size(0)).float().view(-1, 1, 1).expand(bboxes.size(0), bboxes.size(1), 1)
        rois = torch.cat([batch_index, bboxes], dim=-1)
        cropped_masks = self.roi_align(masks.unsqueeze(0), rois[0])
        cropped_masks = cropped_masks[torch.arange(cropped_masks.size(0)), torch.arange(cropped_masks.size(0))]
        return (cropped_masks > 0).unsqueeze(0)

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Export the model."""
        b, _, h, w = batch_inputs.size()
        if b != 1:
            msg = "Only support batch size 1 for export"
            raise ValueError(msg)

        features = self.backbone(batch_inputs)
        outputs, _ = self.sem_seg_head(features)
        mask_cls = outputs["pred_logits"][0]
        mask_pred = outputs["pred_masks"][0]
        pred_boxes = outputs["pred_boxes"][0]

        num_classes = self.sem_seg_head.num_classes
        num_queries = self.num_queries
        test_topk_per_image = self.test_topk_per_image

        scores = mask_cls.sigmoid()
        labels = torch.arange(num_classes).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

        scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes

        mask_pred = torch.nn.functional.interpolate(
            mask_pred.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0]

        mask_pred = mask_pred[topk_indices]
        pred_boxes = pred_boxes[topk_indices]
        pred_masks = (mask_pred > 0).float()

        # Calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
            pred_masks.flatten(1).sum(1) + 1e-6
        )
        pred_scores = scores_per_image * mask_scores_per_image
        pred_classes = labels_per_image.unsqueeze(0)
        pred_boxes = pred_boxes.new_tensor([[w, h, w, h]]) * box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        pred_masks = self.roi_mask_extraction(pred_boxes, pred_masks)

        boxes_with_scores = torch.cat([pred_boxes, pred_scores[:, None]], dim=1)
        boxes_with_scores = boxes_with_scores.unsqueeze(0)

        return (
            boxes_with_scores,
            pred_classes,
            pred_masks,
        )
