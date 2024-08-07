# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for keypoint detection model entity used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from otx.algo.keypoint_detection.utils.data_sample import PoseDataSample
from otx.algo.utils.mmengine_utils import load_checkpoint
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity, KeypointDetBatchPredEntity
from otx.core.metrics import MetricCallable, MetricInput
from otx.core.metrics.pck import PCKMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn


class OTXKeypointDetectionModel(OTXModel[KeypointDetBatchDataEntity, KeypointDetBatchPredEntity]):
    """Base class for the detection models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = PCKMeasureCallable,
        torch_compile: bool = False,
    ) -> None:
        self.image_size = (1, 3, 192, 256)
        self.mean = (0.0, 0.0, 0.0)
        self.std = (255.0, 255.0, 255.0)
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

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
        keypoints_x_label = []
        keypoints_y_label = []
        scores = []
        for output in outputs:
            if not isinstance(output, PoseDataSample):
                raise TypeError(output)
            keypoints.append(torch.as_tensor(output.keypoints, device=self.device))
            scores.append(torch.as_tensor(output.keypoint_weights, device=self.device))
            keypoints_x_label.append(torch.as_tensor(output.keypoint_x_labels, device=self.device))
            keypoints_y_label.append(torch.as_tensor(output.keypoint_y_labels, device=self.device))

        return KeypointDetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            keypoints=keypoints,
            keypoint_x_labels=keypoints_x_label,
            keypoint_y_labels=keypoints_y_label,
            scores=scores,
            keypoints_visible=[],
            bboxes=[],
            labels=[],
            keypoint_weights=[],
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: KeypointDetBatchPredEntity,
        inputs: KeypointDetBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": [
                {
                    "keypoints": kpt.data,
                    "keypoint_x_labels": kpt_x.data,
                    "keypoint_y_labels": kpt_y.data,
                    "scores": scores.data,
                }
                for kpt, kpt_x, kpt_y, scores in zip(
                    preds.keypoints,
                    preds.keypoint_x_labels,
                    preds.keypoint_y_labels,
                    preds.scores,
                )
            ],
            "target": [
                {
                    "keypoints": kpt.data,
                    "keypoint_x_labels": kpt_x.data,
                    "keypoint_y_labels": kpt_y.data,
                    "keypoint_weights": kpt_w.data,
                }
                for kpt, kpt_x, kpt_y, kpt_w in zip(
                    inputs.keypoints,
                    inputs.keypoint_x_labels,
                    inputs.keypoint_y_labels,
                    inputs.keypoint_weights,
                )
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
            model_type="ssd",
            task_type="detection",
            confidence_threshold=self.hparams.get("best_confidence_threshold", None),
            iou_threshold=0.5,
            tile_config=self.tile_config if self.tile_config.enable_tiler else None,
        )
