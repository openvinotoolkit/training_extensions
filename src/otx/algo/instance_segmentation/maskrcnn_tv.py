# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""TV MaskRCNN model implementations."""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn
from torchvision import tv_tensors
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers
from torchvision.models.detection.faster_rcnn import FastRCNNConvFCHead, FastRCNNPredictor, RPNHead, _default_anchorgen
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN,
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNNHeads,
    MaskRCNNPredictor,
)
from torchvision.models.detection.roi_heads import paste_masks_in_image
from torchvision.models.resnet import resnet50

from otx.algo.instance_segmentation.heads import TVRoIHeads
from otx.core.config.data import TileConfig
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable
    from otx.core.schedulers import LRSchedulerListCallable
    from otx.core.types.label import LabelInfoTypes


class _TVMaskRCNN(MaskRCNN):
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


class TVMaskRCNN(ExplainableOTXInstanceSegModel):
    """Implementation of torchvision MaskRCNN for instance segmentation."""

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)
        return {"entity": entity}

    def _customize_outputs(
        self,
        outputs: dict | list[dict],  # type: ignore[override]
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for loss_name, loss_value in outputs.items():
                if isinstance(loss_value, Tensor):
                    losses[loss_name] = loss_value
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            # pop acc from losses
            losses.pop("acc", None)
            return losses

        scores: list[Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        for img_info, prediction in zip(inputs.imgs_info, outputs):
            scores.append(prediction["scores"])
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction["boxes"],
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
            )
            output_masks = tv_tensors.Mask(
                prediction["masks"],
                dtype=torch.bool,
            )
            masks.append(output_masks)
            labels.append(prediction["labels"])

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_map = outputs["saliency_map"].detach().cpu().numpy()
            feature_vector = outputs["feature_vector"].detach().cpu().numpy()

            return InstanceSegBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                polygons=[],
                labels=labels,
                saliency_map=list(saliency_map),
                feature_vector=list(feature_vector),
            )

        return InstanceSegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        if self.input_size is None:
            msg = f"Input size attribute is not set for {self.__class__}"
            raise ValueError(msg)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 3, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="fit_to_window",
            pad_value=0,
            swap_rgb=False,
            via_onnx=True,
            onnx_export_configuration={
                "input_names": ["image"],
                "output_names": ["boxes", "labels", "masks"],
                "dynamic_axes": {
                    "image": {0: "batch", 2: "height", 3: "width"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "masks": {0: "batch", 1: "num_dets", 2: "height", 3: "width"},
                },
                "opset_version": 11,
                "autograd_inlining": False,
            },
            output_names=["bboxes", "labels", "masks"],
        )

    def forward_for_tracing(self, inputs: Tensor) -> tuple[Tensor, ...]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "image_shape": shape,
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(inputs, meta_info_list)


class TVMaskRCNNR50(TVMaskRCNN):
    """Torchvision MaskRCNN model with ResNet50 backbone."""

    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int] = (1024, 1024),
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _create_model(self) -> nn.Module:
        """From torchvision tutorial."""
        # NOTE: Add 1 to num_classes to account for background class.
        num_classes = self.label_info.num_classes + 1

        weights = MaskRCNN_ResNet50_FPN_V2_Weights.verify("DEFAULT")

        trainable_backbone_layers = _validate_trainable_layers(
            is_trained=True,
            trainable_backbone_layers=None,
            max_value=5,
            default_value=3,
        )

        backbone = resnet50(progress=True)
        backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, norm_layer=nn.BatchNorm2d)
        rpn_anchor_generator = _default_anchorgen()
        rpn_head = RPNHead(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7),
            [256, 256, 256, 256],
            [1024],
            norm_layer=nn.BatchNorm2d,
        )
        mask_head = MaskRCNNHeads(backbone.out_channels, [256, 256, 256, 256], 1, norm_layer=nn.BatchNorm2d)

        model = _TVMaskRCNN(
            backbone,
            num_classes=91,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            box_head=box_head,
            mask_head=mask_head,
        )

        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))

        # Replace RoIHeads since torchvision does not allow customized roi_heads.
        model.roi_heads = TVRoIHeads(
            model.roi_heads.box_roi_pool,
            model.roi_heads.box_head,
            model.roi_heads.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=model.roi_heads.score_thresh,
            nms_thresh=model.roi_heads.nms_thresh,
            detections_per_img=model.roi_heads.detections_per_img,
            mask_roi_pool=model.roi_heads.mask_roi_pool,
            mask_head=model.roi_heads.mask_head,
            mask_predictor=model.roi_heads.mask_predictor,
        )

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels

        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes,
        )

        return model
