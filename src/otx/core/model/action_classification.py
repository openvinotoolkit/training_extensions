# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for action_classification model entity used in OTX."""

# mypy: disable-error-code="attr-defined"

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from otx.algo.action_classification.utils.data_sample import ActionDataSample
from otx.core.data.entity.action_classification import ActionClsBatchDataEntity, ActionClsBatchPredEntity
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import MultiClassClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models.utils import ClassificationResult
    from torch import Tensor

    from otx.core.exporter.base import OTXModelExporter
    from otx.core.metrics import MetricCallable


class OTXActionClsModel(OTXModel[ActionClsBatchDataEntity, ActionClsBatchPredEntity]):
    """Base class for the action classification models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.mean = (0.0, 0.0, 0.0)
        self.std = (255.0, 255.0, 255.0)
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.input_size: tuple[int, int]

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

    def _customize_inputs(self, entity: ActionClsBatchDataEntity) -> dict[str, Any]:
        """Convert ActionClsBatchDataEntity into mmaction model's input."""
        mmaction_inputs: dict[str, Any] = {}

        mmaction_inputs["inputs"] = entity.images
        mmaction_inputs["data_samples"] = [
            ActionDataSample(
                metainfo={
                    "img_id": img_info.img_idx,
                    "img_shape": img_info.img_shape,
                    "ori_shape": img_info.ori_shape,
                    "scale_factor": img_info.scale_factor,
                },
                gt_label=labels,
            )
            for img_info, labels in zip(entity.imgs_info, entity.labels)
        ]

        mmaction_inputs["mode"] = "loss" if self.training else "predict"
        return mmaction_inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: ActionClsBatchDataEntity,
    ) -> ActionClsBatchPredEntity | OTXBatchLossEntity:
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
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            input_size=(1, 1, 3, 8, *self.input_size),
            mean=self.mean,
            std=self.std,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=None,
        )

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model(inputs=image, mode="tensor")

    def get_classification_layers(self, prefix: str = "model.") -> dict[str, dict[str, int]]:
        """Get final classification layer information for incremental learning case."""
        sample_model_dict = self._build_model(num_classes=5).state_dict()
        incremental_model_dict = self._build_model(num_classes=6).state_dict()

        classification_layers = {}
        for key in sample_model_dict:
            if sample_model_dict[key].shape != incremental_model_dict[key].shape:
                sample_model_dim = sample_model_dict[key].shape[0]
                incremental_model_dim = incremental_model_dict[key].shape[0]
                stride = incremental_model_dim - sample_model_dim
                num_extra_classes = 6 * sample_model_dim - 5 * incremental_model_dim
                classification_layers[prefix + key] = {"stride": stride, "num_extra_classes": num_extra_classes}
        return classification_layers

    def get_dummy_input(self, batch_size: int = 1) -> ActionClsBatchDataEntity:
        """Returns a dummy input for action classification model."""
        images = torch.rand(batch_size, 1, 3, 8, *self.input_size)
        labels = [torch.LongTensor([0])] * batch_size
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return ActionClsBatchDataEntity(batch_size, images, infos, labels=labels)


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
        return self.model.preprocess(vid)[0][self.model.image_blob_name]

    @property
    def model_adapter_parameters(self) -> dict:
        """Model parameters for export."""
        return {"input_layouts": "NSCTHW"}

    def get_dummy_input(self, batch_size: int = 1) -> ActionClsBatchDataEntity:
        """Returns a dummy input for action classification OV model."""
        # Resize is embedded to the OV model, which means we don't need to know the actual size
        images = [torch.rand(8, 3, 224, 224) for _ in range(batch_size)]
        labels = [torch.LongTensor([0])] * batch_size
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return ActionClsBatchDataEntity(batch_size, images, infos, labels=labels)
