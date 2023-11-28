# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from torchvision import tv_tensors

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmseg.models.data_preprocessor import SegDataPreProcessor
    from omegaconf import DictConfig
    from torch import nn


class OTXSegmentationModel(OTXModel[SegBatchDataEntity, SegBatchPredEntity]):
    """Base class for the detection models used in OTX."""

# This is an example for MMDetection models
# In this way, we can easily import some models developed from the MM community
class MMSegCompatibleModel(OTXSegmentationModel):
    """Segmentation model compatible for MMSeg.

    It can consume MMSeg model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX detection model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        super().__init__()

    def _create_model(self) -> nn.Module:
        from mmseg.registry import MODELS
        try:
            model = MODELS.build(convert_conf_to_mmconfig_dict(self.config, to="tuple"))
        except AssertionError:
            model = MODELS.build(convert_conf_to_mmconfig_dict(self.config, to="list"))
        breakpoint()
        return model

    def _customize_inputs(self, entity: SegBatchDataEntity) -> dict[str, Any]:
        from mmseg.structures import SegDataSample
        from mmengine.structures import PixelData

        mmseg_inputs: dict[str, Any] = {}

        mmseg_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmseg_inputs["data_samples"] = [
            SegDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_sem_seg=PixelData(
                    sem_seg=masks
                ),
            )
            for img_info, masks in zip(
                entity.imgs_info,
                entity.masks,
            )
        ]
        preprocessor: SegDataPreProcessor = self.model.data_preprocessor
        # Don't know why but data_preprocessor.device is not automatically
        # converted by the pl.Trainer's instruction unless the model parameters.
        # Therefore, we change it here in that case.
        if preprocessor.device != (
            model_device := next(self.model.parameters()).device
        ):
            preprocessor = preprocessor.to(device=model_device)
            self.model.data_preprocessor = preprocessor

        mmseg_inputs = preprocessor(data=mmseg_inputs, training=self.training)

        mmseg_inputs["mode"] = "loss" if self.training else "predict"

        return mmseg_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: SegBatchDataEntity,
    ) -> Union[SegBatchPredEntity, OTXBatchLossEntity]:
        from mmseg.structures import SegDataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = sum(v)
            return losses

        masks = []
        scores = []

        for output in outputs:
            if not isinstance(output, SegDataSample):
                raise TypeError(output)

            scores.append(output.pred_instances.scores)
            masks.append(output.pred_instances.masks)

        return SegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            masks=masks
        )
