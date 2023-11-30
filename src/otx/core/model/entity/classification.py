# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

from mmengine.runner import load_checkpoint

# classification.model.backbones should be initialized to register the backbones.
import otx.algo.classification.model.backbones  # noqa: F401
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmpretrain.models.utils import ClsDataPreprocessor
    from omegaconf import DictConfig
    from torch import nn


class OTXClassificationModel(OTXModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity]):
    """Base class for the classification models used in OTX."""

class MMPretrainCompatibleModel(OTXClassificationModel):
    """Classification model compatible for MMPretrain.

    It can consume MMPretrain model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX classification model
    compatible for OTX pipelines.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__()

    def _create_model(self) -> nn.Module:
        from mmpretrain.registry import MODELS
        ### DataPreprocessor config should be converted as tuple
        ### If not, it MODELS.build() creates the empty data preprocessor
        ### That's the reason why I added the lines below
        data_preprocessor_cfg = self.config.pop("data_preprocessor")
        converted_data_preprocessor_cfg = convert_conf_to_mmconfig_dict(
            data_preprocessor_cfg, to="list")
        try:
            converted_cfg = convert_conf_to_mmconfig_dict(self.config, to="tuple")
            converted_cfg.data_preprocessor = converted_data_preprocessor_cfg.to_dict()
            model = MODELS.build(converted_cfg)
        except AssertionError:
            converted_cfg = convert_conf_to_mmconfig_dict(self.config, to="list")
            converted_cfg.data_preprocessor = converted_data_preprocessor_cfg.to_dict()
            model = MODELS.build(converted_cfg)

        if self.load_from is not None:
            load_checkpoint(model, self.load_from)

        return model

    def _customize_inputs(self, entity: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        from mmpretrain.structures import DataSample

        mmpretrain_inputs: dict[str, Any] = {}

        mmpretrain_inputs["inputs"] = entity.images  # B x C x H x W PyTorch tensor
        mmpretrain_inputs["data_samples"] = [
            DataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_label=labels,
            )
            for img_info, labels in zip(
                entity.imgs_info,
                entity.labels,
            )
        ]
        preprocessor: ClsDataPreprocessor = self.model.data_preprocessor
        # Don't know why but data_preprocessor.device is not automatically
        # converted by the pl.Trainer's instruction unless the model parameters.
        # Therefore, we change it here in that case.
        if preprocessor.device != (
            model_device := next(self.model.parameters()).device
        ):
            preprocessor = preprocessor.to(device=model_device)
            self.model.data_preprocessor = preprocessor

        mmpretrain_inputs = preprocessor(data=mmpretrain_inputs, training=self.training)

        mmpretrain_inputs["mode"] = "loss" if self.training else "predict"
        return mmpretrain_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> Union[MulticlassClsBatchPredEntity, OTXBatchLossEntity]:
        from mmpretrain.structures import DataSample
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        scores = []
        labels = []

        for output in outputs:
            if not isinstance(output, DataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )
