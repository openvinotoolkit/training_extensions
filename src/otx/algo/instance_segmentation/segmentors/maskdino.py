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

import torch
from torch import Tensor, nn
from torchvision import tv_tensors
from torchvision.models.detection.image_list import ImageList
from torchvision.ops.roi_align import RoIAlign

from otx.algo.instance_segmentation.heads.maskdino_head import MaskDINOHead
from otx.algo.instance_segmentation.utils import box_ops
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity
from otx.core.utils.mask_util import polygon_to_bitmap


class MaskDINO(nn.Module):
    """Main class for mask classification semantic segmentation architectures."""

    def __init__(
        self,
        backbone: nn.Module,
        sem_seg_head: MaskDINOHead,
        criterion: nn.Module,
        num_queries: int = 300,
        test_topk_per_image: int = 100,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.test_topk_per_image = test_topk_per_image
        self.roi_align = RoIAlign(
            output_size=(28, 28),
            sampling_ratio=0,
            aligned=True,
            spatial_scale=1.0,
        )

    def forward(
        self,
        entity: InstanceSegBatchDataEntity,
        mode: str = "tensor",
    ) -> dict[str, Tensor]:
        """Forward pass."""
        img_shapes = [img_info.img_shape for img_info in entity.imgs_info]
        images = ImageList(entity.images, img_shapes)

        features = self.backbone(images.tensors)

        if self.training:
            targets = []
            for img_info, bboxes, labels, polygons, gt_masks in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
                entity.polygons,
                entity.masks,
            ):
                masks = polygon_to_bitmap(polygons, *img_info.img_shape) if len(polygons) else gt_masks
                norm_shape = torch.tile(torch.tensor(img_info.img_shape, device=img_info.device), (2,))
                targets.append(
                    {
                        "boxes": box_ops.box_xyxy_to_cxcywh(bboxes) / norm_shape,
                        "labels": labels,
                        "masks": tv_tensors.Mask(masks, device=img_info.device, dtype=torch.bool),
                    },
                )

            outputs, mask_dict = self.sem_seg_head(features, targets=targets)
            losses = self.criterion(outputs, targets, mask_dict)
            for k in list(losses.keys()):
                losses[k] *= self.criterion.weight_dict[k]
            return losses

        outputs, _ = self.sem_seg_head(features)
        return outputs

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
        pred_boxes = pred_boxes.new_tensor([[w, h, w, h]]) * box_ops.box_cxcywh_to_xyxy(pred_boxes)
        pred_masks = self.roi_mask_extraction(pred_boxes, pred_masks)

        boxes_with_scores = torch.cat([pred_boxes, pred_scores[:, None]], dim=1)
        boxes_with_scores = boxes_with_scores.unsqueeze(0)

        return (
            boxes_with_scores,
            pred_classes,
            pred_masks,
        )
