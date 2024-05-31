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
from torchvision.models.detection.transform import resize_boxes

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

        detections, detector_losses = self.roi_heads(features, proposals, image_list.image_sizes, targets)

        if not self.training:
            detections = self.postprocess(
                detections,
                image_list.image_sizes,
                ori_shapes,
            )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return self.eager_outputs(losses, detections)

    def postprocess(
        self,
        result: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
        original_image_sizes: list[tuple[int, int]],
        mask_thr_binary: float = 0.5,
    ) -> list[dict[str,  torch.Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
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
        return self.roi_heads.export(features, proposals, image_list.image_sizes)
