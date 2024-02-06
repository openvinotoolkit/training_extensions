# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for instance segmentation model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torchvision import tv_tensors

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import (
    InstanceSegBatchDataEntity,
    InstanceSegBatchPredEntity,
)
from otx.core.data.entity.tile import TileBatchInstSegDataEntity
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.tile_merge import InstanceSegTileMerge

if TYPE_CHECKING:
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import InstanceSegmentationResult
    from torch import nn


class OTXInstanceSegModel(
    OTXModel[InstanceSegBatchDataEntity, InstanceSegBatchPredEntity, TileBatchInstSegDataEntity],
):
    """Base class for the Instance Segmentation models used in OTX."""

    def forward_tiles(self, inputs: TileBatchInstSegDataEntity) -> InstanceSegBatchPredEntity:
        """Unpack instance segmentation tiles.

        Args:
            inputs (TileBatchInstSegDataEntity): Tile batch data entity.

        Returns:
            InstanceSegBatchPredEntity: Merged instance segmentation prediction.
        """
        tile_preds: list[InstanceSegBatchPredEntity] = []
        tile_attrs: list[list[dict[str, int | str]]] = []
        merger = InstanceSegTileMerge(inputs.imgs_info)
        for batch_tile_attrs, batch_tile_input in inputs.unbind():
            output = self.forward(batch_tile_input)
            if isinstance(output, OTXBatchLossEntity):
                msg = "Loss output is not supported for tile merging"
                raise TypeError(msg)
            tile_preds.append(output)
            tile_attrs.append(batch_tile_attrs)
        pred_entities = merger.merge(tile_preds, tile_attrs)

        return InstanceSegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=[pred_entity.image for pred_entity in pred_entities],
            imgs_info=[pred_entity.img_info for pred_entity in pred_entities],
            scores=[pred_entity.score for pred_entity in pred_entities],
            bboxes=[pred_entity.bboxes for pred_entity in pred_entities],
            labels=[pred_entity.labels for pred_entity in pred_entities],
            masks=[pred_entity.masks for pred_entity in pred_entities],
            polygons=[pred_entity.polygons for pred_entity in pred_entities],
        )


class ExplainableOTXInstanceSegModel(OTXInstanceSegModel):
    """OTX Instance Segmentation model which can attach a XAI hook."""

    def register_explain_hook(self) -> None:
        """Register explain hook at the model backbone output."""
        from otx.algo.hooks.recording_forward_hook import MaskRCNNRecordingForwardHook

        self.explain_hook = MaskRCNNRecordingForwardHook.create_and_register_hook(
            num_classes=self.num_classes,
        )

    def remove_explain_hook_handle(self) -> None:
        """Removes explain hook from the model."""
        if self.explain_hook.handle is not None:
            self.explain_hook.handle.remove()

    def reset_explain_hook(self) -> None:
        """Clear all history of explain records."""
        self.explain_hook.reset()


class MMDetInstanceSegCompatibleModel(ExplainableOTXInstanceSegModel):
    """Instance Segmentation model compatible for MMDet."""

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = self.config.pop("load_from", None)
        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        from .utils.mmdet import create_model

        model, self.classification_layers = create_model(self.config, self.load_from)
        return model

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        from mmdet.structures import DetDataSample
        from mmdet.structures.mask import BitmapMasks, PolygonMasks
        from mmengine.structures import InstanceData

        mmdet_inputs: dict[str, Any] = {}

        mmdet_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmdet_inputs["data_samples"] = []

        for img_info, bboxes, masks, polygons, labels in zip(
            entity.imgs_info,
            entity.bboxes,
            entity.masks,
            entity.polygons,
            entity.labels,
        ):
            # NOTE: ground-truth masks are resized in training, but not in inference
            height, width = img_info.img_shape if self.training else img_info.ori_shape
            if len(masks):
                mmdet_masks = BitmapMasks(masks.data.cpu().numpy(), height, width)
            else:
                mmdet_masks = PolygonMasks(
                    [[np.array(polygon.points)] for polygon in polygons],
                    height,
                    width,
                )

            data_sample = DetDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                    "ignored_labels": img_info.ignored_labels,
                },
                gt_instances=InstanceData(
                    bboxes=bboxes,
                    masks=mmdet_masks,
                    labels=labels,
                ),
            )
            mmdet_inputs["data_samples"].append(data_sample)

        preprocessor: DetDataPreprocessor = self.model.data_preprocessor

        mmdet_inputs = preprocessor(data=mmdet_inputs, training=self.training)

        mmdet_inputs["mode"] = "loss" if self.training else "predict"

        return mmdet_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        from mmdet.structures import DetDataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for loss_name, loss_value in outputs.items():
                if isinstance(loss_value, torch.Tensor):
                    losses[loss_name] = loss_value
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            return losses

        scores: list[torch.Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        for output in outputs:
            if not isinstance(output, DetDataSample):
                raise TypeError(output)

            scores.append(output.pred_instances.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    output.pred_instances.bboxes,
                    format="XYXY",
                    canvas_size=output.img_shape,
                ),
            )
            output_masks = tv_tensors.Mask(
                output.pred_instances.masks,
                dtype=torch.bool,
            )
            masks.append(output_masks)
            labels.append(output.pred_instances.labels)

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


class OVInstanceSegmentationModel(
    OVModel[InstanceSegBatchDataEntity, InstanceSegBatchPredEntity],
):
    """Instance segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def _customize_outputs(
        self,
        outputs: list[InstanceSegmentationResult],
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        # add label index
        bboxes = []
        scores = []
        labels = []
        masks = []
        for output in outputs:
            output_objects = output.segmentedObjects
            if len(output_objects):
                bbox = [[output.xmin, output.ymin, output.xmax, output.ymax] for output in output_objects]
            else:
                bbox = torch.empty(size=(0, 0))
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    bbox,
                    format="XYXY",
                    canvas_size=inputs.imgs_info[-1].img_shape,
                ),
            )
            scores.append(torch.tensor([output.score for output in output_objects]))
            masks.append(torch.tensor([output.mask for output in output_objects]))
            labels.append(torch.tensor([output.id for output in output_objects]))

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
