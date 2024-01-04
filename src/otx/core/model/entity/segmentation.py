# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.segmentation import SegBatchDataEntity, SegBatchPredEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.build import build_mm_model

if TYPE_CHECKING:
    from mmseg.models.data_preprocessor import SegDataPreProcessor
    from torch import device, nn


class OTXSegmentationModel(OTXModel[SegBatchDataEntity, SegBatchPredEntity]):
    """Base class for the detection models used in OTX."""


class MMSegCompatibleModel(OTXSegmentationModel):
    """Segmentation model compatible for MMSeg.

    It can consume MMSeg model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX detection model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig | dict) -> None:
        self.load_from = config.pop("load_from", None)
        self.config = DictConfig(config)
        super().__init__()

    def _create_model(self) -> nn.Module:
        from mmengine.registry import MODELS as MMENGINE_MODELS
        from mmseg.models.data_preprocessor import SegDataPreProcessor as _SegDataPreProcessor
        from mmseg.registry import MODELS

        # NOTE: For the history of this monkey patching, please see
        # https://github.com/openvinotoolkit/training_extensions/issues/2743
        @MMENGINE_MODELS.register_module(force=True)
        class SegDataPreProcessor(_SegDataPreProcessor):
            @property
            def device(self) -> device:
                try:
                    buf = next(self.buffers())
                except StopIteration:
                    return super().device
                else:
                    return buf.device

        return build_mm_model(self.config, MODELS, self.load_from)

    def _customize_inputs(self, entity: SegBatchDataEntity) -> dict[str, Any]:
        from mmengine.structures import PixelData
        from mmseg.structures import SegDataSample

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
                    data=masks,
                ),
            )
            for img_info, masks in zip(
                entity.imgs_info,
                entity.masks,
            )
        ]
        preprocessor: SegDataPreProcessor = self.model.data_preprocessor

        mmseg_inputs = preprocessor(data=mmseg_inputs, training=self.training)
        mmseg_inputs["mode"] = "loss" if self.training else "predict"

        return mmseg_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: SegBatchDataEntity,
    ) -> SegBatchPredEntity | OTXBatchLossEntity:
        from mmseg.structures import SegDataSample

        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if "loss" in k:
                    losses[k] = v
            return losses

        masks = []

        for output in outputs:
            if not isinstance(output, SegDataSample):
                raise TypeError(output)
            masks.append(output.pred_sem_seg.data)

        return SegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=masks,
        )
