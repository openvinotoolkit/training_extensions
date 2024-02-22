# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for action_classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from otx.core.data.entity.action_classification import (
    ActionClsBatchDataEntity,
    ActionClsBatchPredEntity,
    ActionClsBatchPredEntityWithXAI,
)
from otx.core.data.entity.base import OTXBatchLossEntity, T_OTXBatchPredEntityWithXAI
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import ClassificationResult
    from torch import nn

    from otx.core.exporter.base import OTXModelExporter


class OTXActionClsModel(
    OTXModel[ActionClsBatchDataEntity, ActionClsBatchPredEntity, T_OTXBatchPredEntityWithXAI, T_OTXTileBatchDataEntity],
):
    """Base class for the action classification models used in OTX."""

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        parameters = super()._export_parameters
        parameters["metadata"].update(
            {
                ("model_info", "model_type"): "Action Classification",
                ("model_info", "task_type"): "action classification",
            },
        )
        return parameters


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
        self.image_size = (1, 1, 3, 8, 224, 224)
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

    @property
    def _export_parameters(self) -> dict[str, Any]:
        """Defines parameters required to export a particular model implementation."""
        export_params = super()._export_parameters
        export_params.update(get_mean_std_from_data_processing(self.config))
        export_params["resize_mode"] = "standard"
        export_params["pad_value"] = 0
        export_params["swap_rgb"] = False
        export_params["via_onnx"] = False
        export_params["input_size"] = self.image_size
        export_params["onnx_export_configuration"] = None

        return export_params

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(**self._export_parameters)


class OVActionClsModel(
    OVModel[ActionClsBatchDataEntity, ActionClsBatchPredEntity, ActionClsBatchPredEntityWithXAI],
):
    """Action Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        model_type: str = "Action Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = False,
        model_api_configuration: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            num_classes,
            model_name,
            model_type,
            async_inference,
            max_num_requests,
            use_throughput_mode,
            model_api_configuration,
        )

    def _customize_inputs(self, entity: ActionClsBatchDataEntity) -> dict[str, Any]:
        # restore original numpy image
        images = [np.transpose(im.cpu().numpy(), (0, 2, 3, 1)) for im in entity.images]
        return {"inputs": images}

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: ActionClsBatchDataEntity,
    ) -> ActionClsBatchPredEntity:
        pred_labels = [torch.tensor(out.top_labels[0][0], dtype=torch.long) for out in outputs]
        pred_scores = [torch.tensor(out.top_labels[0][2]) for out in outputs]

        return ActionClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=pred_labels,
        )

    @property
    def model_parameters(self) -> dict:
        """Model parameters for export."""
        return {"input_layouts": "?NCTHW"}
