# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""TV MaskRCNN model implementations."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor, nn
from torchvision import tv_tensors
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, _default_anchorgen
from torchvision.models.detection.mask_rcnn import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNNPredictor,
)

from otx.algo.instance_segmentation.heads import TVRoIHeads
from otx.algo.instance_segmentation.segmentors.maskrcnn_tv import (
    FastRCNNConvFCHead,
    MaskRCNN,
    MaskRCNNBackbone,
    MaskRCNNHeads,
    RPNHead,
)
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.data.entity.utils import stack_batch
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel


class MaskRCNNTV(ExplainableOTXInstanceSegModel):
    """Implementation of torchvision MaskRCNN for instance segmentation."""

    load_from: ClassVar[dict[str, Any]] = {"maskrcnn_resnet_50": MaskRCNN_ResNet50_FPN_V2_Weights.verify("DEFAULT")}
    mean = (123.675, 116.28, 103.53)
    std = (58.395, 57.12, 57.375)

    AVAILABLE_MODEL_VERSIONS: ClassVar[list[str]] = [
        "maskrcnn_resnet_50",
    ]

    def _create_model(self) -> nn.Module:
        """Create MaskRCNN model with TV implementation."""
        if self.model_name not in self.AVAILABLE_MODEL_VERSIONS:
            msg = f"Model version {self.model_name} is not supported."
            raise ValueError(msg)

        # NOTE: Add 1 to num_classes to account for background class.
        num_classes = self.label_info.num_classes + 1
        weights = self.load_from[self.model_name]

        # init model components, model itself and load weights
        rpn_anchor_generator = _default_anchorgen()
        backbone = MaskRCNNBackbone(model_name=self.model_name)
        rpn_head = RPNHead(model_name=self.model_name, anchorgen=rpn_anchor_generator)
        box_head = FastRCNNConvFCHead(model_name=self.model_name)
        mask_head = MaskRCNNHeads(model_name=self.model_name)

        model = MaskRCNN(
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

        # XAI wraps prediction under dictionary with key "predictions"
        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):
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
            output_names=["bboxes", "labels", "masks", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def forward_for_tracing(self, inputs: Tensor) -> tuple[Tensor, ...]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "image_shape": shape,
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(inputs, meta_info_list, explain_mode=self.explain_mode)
