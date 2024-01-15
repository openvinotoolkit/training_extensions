# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torchvision import tv_tensors

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.detection import DetBatchDataEntity, DetBatchPredEntity
from otx.core.data.entity.tile import TileBatchDetDataEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.build import build_mm_model, get_classification_layers
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.tile_merge import merge_detection_tiles

if TYPE_CHECKING:
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from omegaconf import DictConfig
    from torch import device, nn


class OTXDetectionModel(OTXModel[DetBatchDataEntity, DetBatchPredEntity, TileBatchDetDataEntity]):
    """Base class for the detection models used in OTX."""

    def forward_tiles(self, inputs: TileBatchDetDataEntity) -> DetBatchPredEntity:
        """Unpack detection tiles.

        Args:
            inputs (TileBatchDetDataEntity): Tile batch data entity.

        Returns:
            DetBatchPredEntity: Merged detection prediction.
        """
        tile_preds: list[DetBatchPredEntity] = []
        for tile_input in inputs.unbind():
            output = self.forward(tile_input)
            if isinstance(output, OTXBatchLossEntity):
                msg = "Loss output is not supported for tile merging"
                raise TypeError(msg)
            tile_preds.append(output)
        pred_entity = merge_detection_tiles(tile_preds)

        return DetBatchPredEntity(
            batch_size=inputs.batch_size,
            images=[pred_entity.image],
            imgs_info=[pred_entity.img_info],
            scores=[pred_entity.score],
            bboxes=[pred_entity.bboxes],
            labels=[pred_entity.labels],
        )


class MMDetCompatibleModel(OTXDetectionModel):
    """Detection model compatible for MMDet.

    It can consume MMDet model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX detection model
    compatible for OTX pipelines.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
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

    def _customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
        from mmdet.structures import DetDataSample
        from mmengine.structures import InstanceData

        mmdet_inputs: dict[str, Any] = {}

        mmdet_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmdet_inputs["data_samples"] = [
            DetDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_instances=InstanceData(
                    bboxes=bboxes,
                    labels=labels,
                ),
            )
            for img_info, bboxes, labels in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
            )
        ]
        preprocessor: DetDataPreprocessor = self.model.data_preprocessor

        mmdet_inputs = preprocessor(data=mmdet_inputs, training=self.training)

        mmdet_inputs["mode"] = "loss" if self.training else "predict"

        return mmdet_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        from mmdet.structures import DetDataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, torch.Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        scores = []
        bboxes = []
        labels = []

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
            labels.append(output.pred_instances.labels)

        return DetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )


class OVDetectionCompatibleModel(OTXDetectionModel):
    """Object detection model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        self.model_name = config.pop("model_name")
        self.config = config
        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        from openvino.model_api.models import DetectionModel

        model = DetectionModel.create_model(self.model_name)
        if model.get_label_name(0).lower() == "background":
            logging.warning(
                "Background class detected. Labels shift will be applied during inference to match target labeles",
            )
        return model

    def _customize_inputs(self, entity: DetBatchDataEntity) -> dict[str, Any]:
        if entity.batch_size > 1:
            msg = "Only sync inference with batch = 1 is supported for now"
            raise RuntimeError(msg)
        # restore original numpy image
        img = np.transpose(entity.images[-1].numpy(), (1, 2, 0))
        return {"inputs": img}

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: DetBatchDataEntity,
    ) -> DetBatchPredEntity | OTXBatchLossEntity:
        # add label index
        outputs_objects = outputs.objects
        bboxes = [
            tv_tensors.BoundingBoxes(
                [[output.xmin, output.ymin, output.xmax, output.ymax] for output in outputs_objects],
                format="XYXY",
                canvas_size=inputs.imgs_info[-1].img_shape,
            ),
        ]
        scores = [torch.tensor([output.score for output in outputs_objects])]

        if self.model.get_label_name(0) == "background":
            # some OMZ model requires to shift labeles
            labels = [torch.tensor([output.id - 1 for output in outputs_objects])]
        else:
            labels = [torch.tensor([output.id for output in outputs_objects])]

        return DetBatchPredEntity(
            batch_size=1,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )
