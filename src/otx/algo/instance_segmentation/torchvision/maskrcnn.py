"""Torchvision MaskRCNN model with forward method accepting InstanceSegBatchDataEntity."""
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.roi_heads import paste_masks_in_image

if TYPE_CHECKING:
    from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity


class TVMaskRCNN(MaskRCNN):
    """Torchvision MaskRCNN model with forward method accepting InstanceSegBatchDataEntity."""

    def forward(
        self,
        entity: InstanceSegBatchDataEntity,
    ) -> dict[str, torch.Tensor] | list[dict[str, torch.Tensor]]:
        """Overwrite GeneralizedRCNN forward method to accept InstanceSegBatchDataEntity."""
        ori_shapes = [img_info.ori_shape for img_info in entity.imgs_info]
        img_shapes = [img_info.img_shape for img_info in entity.imgs_info]

        image_list = ImageList(entity.images, img_shapes)
        targets = []
        for bboxes, labels, masks, polygons in zip(
            entity.bboxes,
            entity.labels,
            entity.masks,
            entity.polygons,
        ):
            targets.append(
                {
                    "boxes": bboxes,
                    # TODO(Eugene): num_classes + 1 (BG) as torchvision.MaskRCNN assume background class?
                    "labels": labels + 1,
                    "masks": masks,
                    "polygons": polygons,
                },
            )

        features = self.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(image_list, features, targets)

        detections, detector_losses = self.roi_heads(
            features,
            proposals,
            image_list.image_sizes,
            targets,
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses
        scale_factors = [img_meta.scale_factor for img_meta in entity.imgs_info]
        return self.postprocess(
            detections,
            ori_shapes,
            scale_factors,
        )

    def postprocess(
        self,
        result: list[dict[str, torch.Tensor]],
        ori_shapes: list[tuple[int, int]],
        scale_factors: list[tuple[float, float]],
        mask_thr_binary: float = 0.5,
    ) -> list[dict[str, torch.Tensor]]:
        for i, (pred, scale_factor, ori_shape) in enumerate(zip(result, scale_factors, ori_shapes)):
            boxes = pred["boxes"]
            # boxes = resize_boxes(boxes, im_s, o_im_s)
            _scale_factor = [1 / s for s in scale_factor]  # (H, W)
            boxes = boxes * boxes.new_tensor(_scale_factor[::-1]).repeat((1, int(boxes.size(-1) / 2)))

            result[i]["boxes"] = boxes
            result[i]["labels"] -= 1  # Convert back to 0-indexed labels
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, ori_shape)
                masks = (masks >= mask_thr_binary).to(dtype=torch.bool)
                masks = masks.squeeze(1)
                result[i]["masks"] = masks
        return result

    def export(
        self,
        batch_inputs: torch.Tensor,
        batch_img_metas: list[dict],
    ):
        img_shapes = [img_meta["image_shape"] for img_meta in batch_img_metas]
        image_list = ImageList(batch_inputs, img_shapes)
        features = self.backbone(batch_inputs)
        proposals, _ = self.rpn(image_list, features)
        boxes, labels, masks_probs = self.roi_heads.export(features, proposals, image_list.image_sizes)
        labels = [label - 1 for label in labels]  # Convert back to 0-indexed labels
        return boxes, labels, masks_probs
