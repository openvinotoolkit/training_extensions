# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Torchvision MaskRCNN model with forward method accepting InstanceSegBatchDataEntity."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch import Tensor, nn
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead as _FastRCNNConvFCHead
from torchvision.models.detection.faster_rcnn import RPNHead as _RPNHead
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN as _MaskRCNN,
)
from torchvision.models.detection.mask_rcnn import (
    MaskRCNNHeads as _MaskRCNNHeads,
)
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.models.resnet import resnet50

if TYPE_CHECKING:
    from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity


class MaskRCNN(_MaskRCNN):
    """Torchvision MaskRCNN model with forward method accepting InstanceSegBatchDataEntity."""

    def forward(self, entity: InstanceSegBatchDataEntity) -> dict[str, Tensor] | list[dict[str, Tensor]]:
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
            # NOTE: shift labels by 1 as 0 is reserved for background
            _labels = labels + 1 if len(labels) else labels
            targets.append(
                {
                    "boxes": bboxes,
                    "labels": _labels,
                    "masks": masks,
                    "polygons": polygons,
                },
            )

        features = self.backbone(image_list.tensors)
        if isinstance(features, Tensor):
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
        scale_factors = [
            img_meta.scale_factor if img_meta.scale_factor else (1.0, 1.0) for img_meta in entity.imgs_info
        ]

        return self.postprocess(detections, ori_shapes, scale_factors)

    def postprocess(
        self,
        result: list[dict[str, Tensor]],
        ori_shapes: list[tuple[int, int]],
        scale_factors: list[tuple[float, float]],
        mask_thr_binary: float = 0.5,
    ) -> list[dict[str, Tensor]]:
        """Postprocess the output of the model."""
        for i, (pred, scale_factor, ori_shape) in enumerate(zip(result, scale_factors, ori_shapes)):
            boxes = pred["boxes"]
            labels = pred["labels"]
            _scale_factor = [1 / s for s in scale_factor]  # (H, W)
            boxes = boxes * boxes.new_tensor(_scale_factor[::-1]).repeat((1, int(boxes.size(-1) / 2)))
            h, w = ori_shape
            boxes[:, 0::2].clamp_(min=0, max=w - 1)
            boxes[:, 1::2].clamp_(min=0, max=h - 1)
            keep_indices = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0
            boxes = boxes[keep_indices > 0]
            labels = labels[keep_indices > 0]
            result[i]["boxes"] = boxes
            result[i]["labels"] = labels - 1  # Convert back to 0-indexed labels
            if "masks" in pred:
                masks = pred["masks"][keep_indices]
                masks = paste_masks_in_image(masks, boxes, ori_shape)
                masks = (masks >= mask_thr_binary).to(dtype=torch.bool)
                masks = masks.squeeze(1)
                result[i]["masks"] = masks
        return result

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
    ) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """Export the model with the given inputs and image metas."""
        img_shapes = [img_meta["image_shape"] for img_meta in batch_img_metas]
        image_list = ImageList(batch_inputs, img_shapes)
        features = self.backbone(batch_inputs)
        proposals, _ = self.rpn(image_list, features)
        boxes, labels, masks_probs = self.roi_heads.export(features, proposals, image_list.image_sizes)
        labels = [label - 1 for label in labels]  # Convert back to 0-indexed labels
        return boxes, labels, masks_probs


class MaskRCNNBackbone:
    """Implementation of MaskRCNN torchvision factory for instance segmentation."""

    def __new__(cls, model_name: str) -> nn.Module:
        """Create MaskRCNNBackbone."""
        trainable_backbone_layers = _validate_trainable_layers(
            is_trained=True,
            trainable_backbone_layers=None,
            max_value=5,
            default_value=3,
        )
        if model_name == "maskrcnn_resnet_50":
            return _resnet_fpn_extractor(resnet50(progress=True), trainable_backbone_layers, norm_layer=nn.BatchNorm2d)

        msg = "Model name {model_name} is not supported."
        raise ValueError(msg)


class RPNHead:
    """Implementation of RPNHead for MaskRCNN."""

    RPNHEAD_CFG: ClassVar[dict[str, Any]] = {
        "maskrcnn_resnet_50": {
            "in_channels": 256,
            "conv_depth": 2,
        },
    }

    def __new__(cls, model_name: str, anchorgen: AnchorGenerator) -> nn.Module:
        """Create RPNHead."""
        return _RPNHead(**cls.RPNHEAD_CFG[model_name], num_anchors=anchorgen.num_anchors_per_location()[0])


class FastRCNNConvFCHead:
    """Implementation of FastRCNNConvFCHead for MaskRCNN."""

    FASTRCNN_CFG: ClassVar[dict[str, Any]] = {
        "maskrcnn_resnet_50": {
            "input_size": (256, 7, 7),
            "conv_layers": [256, 256, 256, 256],
            "fc_layers": [1024],
        },
    }

    def __new__(cls, model_name: str) -> nn.Module:
        """Create FastRCNNConvFCHead."""
        return _FastRCNNConvFCHead(**cls.FASTRCNN_CFG[model_name], norm_layer=nn.BatchNorm2d)


class MaskRCNNHeads:
    """Implementation of MaskRCNNHeads for MaskRCNN."""

    MASKRCNNHEADS_CFG: ClassVar[dict[str, Any]] = {
        "maskrcnn_resnet_50": {
            "in_channels": 256,
            "layers": [256, 256, 256, 256],
            "dilation": 1,
        },
    }

    def __new__(cls, model_name: str) -> nn.Module:
        """Create MaskRCNNHeads."""
        return _MaskRCNNHeads(**cls.MASKRCNNHEADS_CFG[model_name], norm_layer=nn.BatchNorm2d)
