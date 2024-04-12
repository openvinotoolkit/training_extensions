# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for action_classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from otx.core.data.entity.action_classification import ActionClsBatchDataEntity, ActionClsBatchPredEntity
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.tile import T_OTXTileBatchDataEntity
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import MultiClassClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.utils.config import inplace_num_classes
from otx.core.utils.utils import get_mean_std_from_data_processing

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from omegaconf import DictConfig
    from openvino.model_api.models.utils import ClassificationResult
    from torch import nn

    from otx.core.exporter.base import OTXModelExporter
    from otx.core.metrics import MetricCallable


class OTXActionClsModel(
    OTXModel[
        ActionClsBatchDataEntity,
        ActionClsBatchPredEntity,
        T_OTXTileBatchDataEntity,
    ],
):
    """Base class for the action classification models used in OTX."""

    def __init__(
        self,
        num_classes: int,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Action Classification",
            task_type="action classification",
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: ActionClsBatchPredEntity,
        inputs: ActionClsBatchDataEntity,
    ) -> MetricInput:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }


class MMActionCompatibleModel(OTXActionClsModel):
    """Action classification model compitible for MMAction.

    It can consume MMAction model configuration translated into OTX configuration
    (please see otx.tools.translate_mmrecipe) and create the OTX Action classification model
    compatible for OTX pipelines.
    """

    def __init__(
        self,
        num_classes: int,
        config: DictConfig,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        config = inplace_num_classes(cfg=config, num_classes=num_classes)
        self.config = config
        self.load_from = config.pop("load_from", None)
        self.image_size = (1, 1, 3, 8, 224, 224)
        super().__init__(
            num_classes=num_classes,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

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
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        mean, std = get_mean_std_from_data_processing(self.config)

        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=self.image_size,
            mean=mean,
            std=std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=None,
        )


class OVActionClsModel(
    OVModel[ActionClsBatchDataEntity, ActionClsBatchPredEntity],
):
    """Action Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Action Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = False,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MultiClassClsMetricCallable,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
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

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: ActionClsBatchPredEntity,
        inputs: ActionClsBatchDataEntity,
    ) -> MetricInput:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }

    def transform_fn(self, data_batch: ActionClsBatchDataEntity) -> np.array:
        """Data transform function for PTQ."""
        np_data = self._customize_inputs(data_batch)
        vid = np_data["inputs"][0]
        vid = self.model.preprocess(vid)[0][self.model.image_blob_name]
        return self.model._change_layout(vid)  # noqa: SLF001

    @property
    def model_adapter_parameters(self) -> dict:
        """Model parameters for export."""
        return {"input_layouts": "?NCTHW"}
