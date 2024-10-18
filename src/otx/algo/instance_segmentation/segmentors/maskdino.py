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
from otx.algo.instance_segmentation.losses import MaskDINOCriterion
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
        self.roi_align = RoIAlign(
            output_size=(28, 28),
            sampling_ratio=0,
            aligned=True,
            spatial_scale=1.0,
        )

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
        return cropped_masks > 0

    def calculate_object_scores(self, mask_pred: Tensor) -> Tensor:
        """Calculate object scores from mask prediction.

        Args:
            mask_pred (Tensor): logits of mask prediction

        Returns:
            Tensor: object scores
        """
        pred_masks = (mask_pred > 0).to(mask_pred)

        # Calculate average mask prob
        return (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)

    def forward(
        self,
        features: dict[str, Tensor],
        targets: list[dict[str, Tensor]] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Forward pass."""
        mask_features, _, multi_scale_features = self.pixel_decoder(features)
        return self.predictor(multi_scale_features, mask_features, targets=targets)

    @torch.no_grad()
    def predict(
        self,
        features: dict[str, Tensor],
        imgs_info: list[ImageInfo] | list[dict[str, Any]],
        export: bool = False,
    ) -> tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]:
        """Predict function.

        Args:
            features (dict[str, Tensor]): feature maps
            imgs_info (list[ImageInfo] | list[dict[str, Any]]):
                list[ImageInfo]: image info (i.e ori_shape) list regarding original images used for training
                list[dict[str, Any]]: image info (i.e img_shape) used for exporting
            export (bool, optional): whether to export the model. Defaults to False.

        Returns:
            tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]:
                list[Tensor]: bounding boxes and scores with shape [N, 5]
                list[torch.LongTensor]: labels with shape [N]
                list[tv_tensors.Mask]: masks with shape [N, H, W]
        """
        outputs, _ = self(features)

        class_queries_logits = outputs["pred_logits"]
        masks_queries_logits = outputs["pred_masks"]
        mask_box_results = outputs["pred_boxes"]

        device = masks_queries_logits.device
        num_classes = self.num_classes
        num_queries = self.num_queries
        test_topk_per_image = self.test_topk_per_image

        batch_bboxes_scores: list[Tensor] = []
        batch_labels: list[torch.LongTensor] = []
        batch_masks: list[tv_tensors.Mask] = []

        for mask_pred, mask_cls, pred_boxes, img_info in zip(
            masks_queries_logits,
            class_queries_logits,
            mask_box_results,
            imgs_info,
        ):
            h, w = img_info["img_shape"] if export else img_info.ori_shape  # type: ignore[index, attr-defined]
            scores = mask_cls.sigmoid()
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(test_topk_per_image, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = topk_indices // num_classes

            mask_pred = mask_pred[topk_indices]  # noqa: PLW2901
            pred_boxes = pred_boxes[topk_indices]  # noqa: PLW2901
            pred_classes = labels_per_image

            pred_masks = torch.nn.functional.interpolate(
                mask_pred.unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0]

            pred_scores = scores_per_image * self.calculate_object_scores(pred_masks)
            pred_boxes = pred_boxes.new_tensor([[w, h, w, h]]) * box_convert(  # noqa: PLW2901
                pred_boxes,
                in_fmt="cxcywh",
                out_fmt="xyxy",
            )
            pred_boxes[:, 0::2].clamp_(min=0, max=w - 1)
            pred_boxes[:, 1::2].clamp_(min=0, max=h - 1)

            # Extract masks from RoI for exporting the model
            if export:
                pred_masks = self.roi_mask_extraction(pred_boxes, pred_masks) > 0
                # Create dummy filter as Model API has its own filtering mechanism.
                keep = pred_scores > 0.05
            else:
                pred_masks = pred_masks > 0
                area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
                keep = (pred_masks.sum((1, 2)) > 5) & (area > 10) & (pred_scores > 0.05)

            boxes_scores = torch.cat([pred_boxes, pred_scores[:, None]], dim=1)
            batch_masks.append(pred_masks[keep])
            batch_bboxes_scores.append(boxes_scores[keep])
            batch_labels.append(pred_classes[keep])

        return batch_bboxes_scores, batch_labels, batch_masks


class MaskDINO(nn.Module):
    """Main class for mask classification semantic segmentation architectures.

    Args:
        backbone (nn.Module): backbone network
        sem_seg_head (MaskDINOHead): MaskDINO head including pixel decoder and predictor
        criterion (MaskDINOCriterion): MaskDINO loss criterion
    """

    def __init__(
        self,
        backbone: nn.Module,
        sem_seg_head: MaskDINOHead,
        criterion: MaskDINOCriterion,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion

    def forward(
        self,
        images: ImageList,
        imgs_info: list[ImageInfo],
        targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Tensor] | tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]:
        """Forward pass.

        Args:
            images (ImageList): input images
            imgs_info (list[ImageInfo]): image info (i.e ori_shape) list regarding original images
            targets (list[dict[str, Any]] | None, optional): ground-truth annotations. Defaults to None.

        Returns:
            dict[str, Tensor] | tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]:
                dict[str, Tensor]: loss values
                tuple[list[Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]: prediction results
                    list[Tensor]: bounding boxes and scores with shape [N, 5]
                    list[torch.LongTensor]: labels with shape [N]
                    list[tv_tensors.Mask]: masks with shape [N, H, W]
        """
        features = self.backbone(images.tensors)

        if self.training:
            outputs, mask_dict = self.sem_seg_head(features, targets=targets)
            losses = self.criterion(outputs, targets, mask_dict)
            for k in list(losses.keys()):
                losses[k] *= self.criterion.weight_dict[k]
            return losses

        return self.sem_seg_head.predict(features, imgs_info)

    def export(
        self,
        batch_inputs: torch.Tensor,
        batch_img_metas: list[dict],
    ) -> tuple[list[torch.Tensor], list[torch.LongTensor], list[tv_tensors.Mask]]:
        """Export the model."""
        if len(batch_inputs) != 1:
            msg = "Only support batch size 1 for export"
            raise ValueError(msg)

        features = self.backbone(batch_inputs)
        return self.sem_seg_head.predict(features, batch_img_metas, export=True)
