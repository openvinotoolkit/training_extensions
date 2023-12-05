# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.structures import InstanceData
from torchvision import tv_tensors

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from omegaconf import DictConfig
    from torch import nn


class OTXInstanceSegModel(OTXModel[InstanceSegBatchDataEntity, InstanceSegBatchPredEntity]):
    """Base class for the detection models used in OTX."""


# This is an example for MMDetection models
# In this way, we can easily import some models developed from the MM community
class MMDetInstanceSegCompatibleModel(OTXInstanceSegModel):
    """Detection model compatible for MMDet.

    It can consume MMDet model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX detection model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_from = self.config.pop("load_from", None)
        super().__init__()

    def _create_model(self) -> nn.Module:
        det = MODELS.get("DetDataPreprocessor")
        MMENGINE_MODELS.register_module(module=det)

        try:
            model = MODELS.build(convert_conf_to_mmconfig_dict(self.config, to="tuple"))
        except AssertionError:
            model = MODELS.build(convert_conf_to_mmconfig_dict(self.config, to="list"))

        if self.load_from is not None:
            load_checkpoint(model, self.load_from)

        return model

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
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
            height, width = img_info.ori_shape
            if len(masks):
                mmdet_masks = BitmapMasks(
                    masks.data.cpu().numpy(), height, width)
            else:
                mmdet_masks = PolygonMasks(
                    [[np.array(polygon.points)] for polygon in polygons], height, width)

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
        # Don't know why but data_preprocessor.device is not automatically
        # converted by the pl.Trainer's instruction unless the model parameters.
        # Therefore, we change it here in that case.
        if preprocessor.device != (
            model_device := next(self.model.parameters()).device
        ):
            preprocessor = preprocessor.to(device=model_device)
            self.model.data_preprocessor = preprocessor

        mmdet_inputs = preprocessor(data=mmdet_inputs, training=self.training)

        mmdet_inputs["mode"] = "loss" if self.training else "predict"

        return mmdet_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: InstanceSegBatchDataEntity,
    ) -> Union[InstanceSegBatchPredEntity, OTXBatchLossEntity]:

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

        scores: list[float] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[int] = []
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
            output_masks = tv_tensors.Mask(output.pred_instances.masks, dtype=torch.bool)
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
