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
from otx.core.model.entity.base import OTXModel
from otx.core.utils.build import build_mm_model, get_classification_layers
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.tile_merge import merge_inst_seg_tiles

if TYPE_CHECKING:
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from omegaconf import DictConfig
    from torch import device, nn


class OTXInstanceSegModel(
    OTXModel[InstanceSegBatchDataEntity, InstanceSegBatchPredEntity],
):
    """Base class for the detection models used in OTX."""

    def unpack_inst_seg_tiles(self, inputs: TileBatchInstSegDataEntity) -> InstanceSegBatchPredEntity:
        """Unpack instance segmentation tiles.

        Args:
            inputs (TileBatchInstSegDataEntity): _description_

        Returns:
            InstanceSegBatchPredEntity: _description_
        """
        pred_entities = []
        for tiles, tile_infos, bboxes, masks, polygons, labels in zip(
            inputs.batch_tiles,
            inputs.batch_tile_infos,
            inputs.bboxes,
            inputs.masks,
            inputs.polygons,
            inputs.labels,
        ):
            tile_preds: list[InstanceSegBatchPredEntity] = []
            for tile, tile_info in zip(tiles, tile_infos):
                tile_input = InstanceSegBatchDataEntity(
                    batch_size=1,
                    images=[tile],
                    imgs_info=[tile_info],
                    bboxes=[bboxes],
                    masks=[masks],
                    polygons=[polygons],
                    labels=[labels],
                )
                output = self.forward(tile_input)
                if isinstance(output, OTXBatchLossEntity):
                    msg = "Loss output is not supported for tile merging"
                    raise RuntimeError(msg)
                tile_preds.append(output)
            pred_entities.append(merge_inst_seg_tiles(tile_preds))
        return InstanceSegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=[entity.image for entity in pred_entities],
            imgs_info=[entity.img_info for entity in pred_entities],
            scores=[entity.score for entity in pred_entities],
            bboxes=[entity.bboxes for entity in pred_entities],
            labels=[entity.labels for entity in pred_entities],
            masks=[entity.masks for entity in pred_entities],
            polygons=[entity.polygons for entity in pred_entities],
        )


class MMDetInstanceSegCompatibleModel(OTXInstanceSegModel):
    """Instance Segmentation model compatible for MMDet."""

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = self.config.pop("load_from", None)
        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        from mmdet.models.data_preprocessors import (
            DetDataPreprocessor as _DetDataPreprocessor,
        )
        from mmdet.registry import MODELS
        from mmengine.registry import MODELS as MMENGINE_MODELS

        # NOTE: For the history of this monkey patching, please see
        # https://github.com/openvinotoolkit/training_extensions/issues/2743
        @MMENGINE_MODELS.register_module(force=True)
        class DetDataPreprocessor(_DetDataPreprocessor):
            @property
            def device(self) -> device:
                try:
                    buf = next(self.buffers())
                except StopIteration:
                    return super().device
                else:
                    return buf.device

        self.classification_layers = get_classification_layers(self.config, MODELS, "model.")
        return build_mm_model(self.config, MODELS, self.load_from)

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


class OVInstanceSegCompatibleModel(OTXInstanceSegModel):
    """Instance segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        self.model_name = config.pop("model_name")
        self.model_type = config.pop("model_type")
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        from openvino.model_api.models import Model

        return Model.create_model(self.model_name, self.model_type)

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        if entity.batch_size > 1:
            msg = "Only sync inference with batch = 1 is supported for now"
            raise RuntimeError(msg)
        # restore original numpy image
        img = np.transpose(entity.images[-1].numpy(), (1, 2, 0))
        return {"inputs": img}

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        # add label index
        scores = [torch.as_tensor(outputs[0])]
        labels = [torch.as_tensor(outputs[1])]
        bboxes = [
            tv_tensors.BoundingBoxes(
                outputs[2],
                format="XYXY",
                canvas_size=inputs.imgs_info[-1].img_shape,
            ),
        ]
        masks = [tv_tensors.Mask(outputs[3])]

        return InstanceSegBatchPredEntity(
            batch_size=1,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )
