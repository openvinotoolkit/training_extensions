# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for action_detection model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchvision import tv_tensors

from otx.core.data.entity.action_detection import ActionDetBatchDataEntity, ActionDetBatchPredEntity
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.model.entity.base import OTXModel
from otx.core.utils.build import build_mm_model, get_classification_layers
from otx.core.utils.config import inplace_num_classes

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch import device, nn


class OTXActionDetModel(OTXModel[ActionDetBatchDataEntity, ActionDetBatchPredEntity, T_OTXTileBatchDataEntity]):
    """Base class for the action detection models used in OTX."""


class MMActionCompatibleModel(OTXActionDetModel):
    """Action detection model compitible for MMAction.

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

    def _customize_inputs(self, entity: ActionDetBatchDataEntity) -> dict[str, Any]:
        """Convert ActionClsBatchDataEntity into mmaction model's input."""
        from mmaction.structures import ActionDataSample
        from mmengine.structures import InstanceData

        mmaction_inputs: dict[str, Any] = {}

        mmaction_inputs["inputs"] = entity.images
        entity_proposals = entity.proposals if entity.proposals else [None] * len(entity.bboxes)
        mmaction_inputs["data_samples"] = [
            ActionDataSample(
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
                proposals=InstanceData(bboxes=proposals) if proposals is not None else None,
            )
            for img_info, bboxes, labels, proposals in zip(
                entity.imgs_info,
                entity.bboxes,
                entity.labels,
                entity_proposals,
            )
        ]

        mmaction_inputs = self.model.data_preprocessor(data=mmaction_inputs, training=self.training)
        mmaction_inputs["mode"] = "loss" if self.training else "predict"
        return mmaction_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ActionDetBatchDataEntity,
    ) -> ActionDetBatchPredEntity | OTXBatchLossEntity:
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
        bboxes = []
        proposals = []

        for output in outputs:
            if not isinstance(output, ActionDataSample):
                raise TypeError(output)

            output_scores, output_labels = output.pred_instances.scores.max(-1)
            output.pred_instances.bboxes[:, 0::2] *= output.img_shape[1]
            output.pred_instances.bboxes[:, 1::2] *= output.img_shape[0]
            scores.append(output_scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    output.pred_instances.bboxes,
                    format="XYXY",
                    canvas_size=output.img_shape,
                ),
            )
            labels.append(output_labels)
            proposals.append(output.proposals.bboxes)

        return ActionDetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
            proposals=proposals,
        )
