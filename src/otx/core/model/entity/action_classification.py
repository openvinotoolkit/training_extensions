# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for action_classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.core.data.entity.action_classification import (
    ActionClsBatchDataEntity,
    ActionClsBatchPredEntity,
)
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.build import build_mm_model, get_classification_layers
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import device, nn


class OTXActionClsModel(OTXModel[ActionClsBatchDataEntity, ActionClsBatchPredEntity, T_OTXTileBatchDataEntity]):
    """Base class for the action classification models used in OTX."""


class MMActionCompatibleModel(OTXActionClsModel):
    """Action classification model compitible for MMAction.

    It can consume MMAction model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX Action classification model
    compatible for OTX pipelines.
    """

    def __init__(self, num_classes: int, config: DictConfig) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        super().__init__(num_classes=num_classes)

    def _create_model(self) -> nn.Module:
        from .utils.mmaction import create_model
        model, self.classification_layers = create_model(self.config, self.load_from)
        return model

    def _customize_inputs(self, entity: ActionClsBatchDataEntity) -> dict[str, Any]:
        """Convert ActionClsBatchDataEntity into mmaction model's input."""
        from mmaction.structures import ActionDataSample

        mmaction_inputs: dict[str, Any] = {}

        mmaction_inputs["inputs"] = entity.images
        mmaction_inputs["data_samples"] = [
            ActionDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "pad_shape": img_info.pad_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_label=labels,
            )
            for img_info, labels in zip(entity.imgs_info, entity.labels)
        ]

        mmaction_inputs = self.model.data_preprocessor(data=mmaction_inputs, training=self.training)
        mmaction_inputs["mode"] = "loss" if self.training else "predict"
        return mmaction_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ActionClsBatchDataEntity,
    ) -> ActionClsBatchPredEntity | OTXBatchLossEntity:
        from mmaction.structures import ActionDataSample

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
            if not isinstance(output, ActionDataSample):
                raise TypeError(output)

            scores.append(output.pred_score)
            labels.append(output.pred_label)

        return ActionClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )
