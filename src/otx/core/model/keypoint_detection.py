# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for keypoint detection model entity used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from otx.algo.utils.mmengine_utils import load_checkpoint
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity, KeypointDetBatchPredEntity
from otx.core.metrics import MetricCallable, MetricInput
from otx.core.metrics.pck import PCKMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models.utils import DetectedKeypoints
    from torch import nn


class OTXKeypointDetectionModel(OTXModel[KeypointDetBatchDataEntity, KeypointDetBatchPredEntity]):
    """Base class for the keypoint detection models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        input_size: tuple[int, int],
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = PCKMeasureCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            input_size=input_size,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.input_size: tuple[int, int]

    @abstractmethod
    def _build_model(self, num_classes: int) -> nn.Module:
        raise NotImplementedError

    def _create_model(self) -> nn.Module:
        detector = self._build_model(num_classes=self.label_info.num_classes)
        detector.init_weights()
        self.classification_layers = self.get_classification_layers(prefix="model.")
        if self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")
        return detector

    def _customize_inputs(self, entity: KeypointDetBatchDataEntity) -> dict[str, Any]:
        """Convert KeypointDetBatchDataEntity into Topdown model's input."""
        inputs: dict[str, Any] = {}

        inputs["inputs"] = entity.images
        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"
        return inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: KeypointDetBatchDataEntity,
    ) -> KeypointDetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        keypoints = []
        scores = []
        for output in outputs:
            if not isinstance(output, tuple):
                raise TypeError(output)
            keypoints.append(torch.as_tensor(output[0], device=self.device))
            scores.append(torch.as_tensor(output[1], device=self.device))

        return KeypointDetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            keypoints=keypoints,
            scores=scores,
            keypoints_visible=[],
            bboxes=[],
            labels=[],
            bbox_info=[],
        )

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        self._metric.input_size = self.input_size

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: KeypointDetBatchPredEntity,
        inputs: KeypointDetBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": [
                {
                    "keypoints": kpt,
                    "scores": score,
                }
                for kpt, score in zip(preds.keypoints, preds.scores)
            ],
            "target": [
                {
                    "keypoints": kpt,
                    "keypoints_visible": kpt_visible,
                }
                for kpt, kpt_visible in zip(inputs.keypoints, inputs.keypoints_visible)
            ],
        }

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

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(inputs=image, mode="tensor")

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="keypoint_detection",
            task_type="keypoint_detection",
            confidence_threshold=self.hparams.get("best_confidence_threshold", None),
        )

    def get_dummy_input(self, batch_size: int = 1) -> KeypointDetBatchDataEntity:
        """Returns a dummy input for key point detection model."""
        images = torch.rand(batch_size, 3, *self.input_size)
        return KeypointDetBatchDataEntity(
            batch_size,
            images,
            [],
            [torch.tensor([0, 0, self.input_size[1], self.input_size[0]])],
            labels=[],
            bbox_info=[],
            keypoints=[],
            keypoints_visible=[],
        )


class OVKeypointDetectionModel(OVModel[KeypointDetBatchDataEntity, KeypointDetBatchPredEntity]):
    """Keypoint detection model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX keypoint detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "keypoint_detection",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = PCKMeasureCallable,
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

    def _customize_outputs(
        self,
        outputs: list[DetectedKeypoints],
        inputs: KeypointDetBatchDataEntity,
    ) -> KeypointDetBatchPredEntity | OTXBatchLossEntity:
        keypoints = []
        scores = []
        for output in outputs:
            keypoints.append(torch.as_tensor(output.keypoints, device=self.device))
            scores.append(torch.as_tensor(output.scores, device=self.device))

        return KeypointDetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            keypoints=keypoints,
            scores=scores,
            keypoints_visible=[],
            bboxes=[],
            labels=[],
            bbox_info=[],
        )

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        self._metric.input_size = (self.model.h, self.model.w)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: KeypointDetBatchPredEntity,
        inputs: KeypointDetBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": [
                {
                    "keypoints": kpt,
                    "scores": score,
                }
                for kpt, score in zip(preds.keypoints, preds.scores)
            ],
            "target": [
                {
                    "keypoints": kpt,
                    "keypoints_visible": kpt_visible,
                }
                for kpt, kpt_visible in zip(inputs.keypoints, inputs.keypoints_visible)
            ],
        }
