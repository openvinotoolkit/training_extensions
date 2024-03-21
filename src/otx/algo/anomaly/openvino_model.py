"""OTX Anomaly OpenVINO model.

All anomaly models use the same AnomalyDetection model from ModelAPI.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from anomalib import TaskType as AnomalibTaskType
from anomalib.metrics import create_metric_collection
from anomalib.metrics.threshold import BaseThreshold, F1AdaptiveThreshold
from lightning.pytorch import LightningModule

from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.model.module.anomaly import AnomalyModelInputs
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from anomalib.metrics import AnomalibMetricCollection
    from lightning.pytorch.callbacks.callback import Callback
    from openvino.model_api.models import Model
    from openvino.model_api.models.anomaly import AnomalyResult


class AnomalyOpenVINO(OVModel, OTXModel, LightningModule):
    """Anomaly OpenVINO model."""

    # [TODO](ashwinvaidya17): Remove LightningModule once OTXModel is updated to use LightningModule.
    # NOTE: Ideally OVModel should not be a LightningModule

    def __init__(
        self,
        model_name: str,
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        num_classes: int = 2,
    ) -> None:
        super().__init__(
            num_classes=num_classes,  # NOTE: Ideally this should be set to 2 always
            model_name=model_name,
            model_type="AnomalyDetection",
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
        )
        self.image_threshold: BaseThreshold
        self.pixel_threshold: BaseThreshold

        self.image_metrics: AnomalibMetricCollection
        self.pixel_metrics: AnomalibMetricCollection

    @property
    def task(self) -> AnomalibTaskType:
        """Return the task type of the model."""
        if self._task_type:
            return self._task_type
        msg = "``self._task_type`` is not assigned"
        raise AttributeError(msg)

    @task.setter
    def task(self, value: OTXTaskType) -> None:
        if value == OTXTaskType.ANOMALY_CLASSIFICATION:
            self._task_type = AnomalibTaskType.CLASSIFICATION
        elif value == OTXTaskType.ANOMALY_DETECTION:
            self._task_type = AnomalibTaskType.DETECTION
        elif value == OTXTaskType.ANOMALY_SEGMENTATION:
            self._task_type = AnomalibTaskType.SEGMENTATION
        else:
            msg = f"Unexpected task type: {value}"
            raise ValueError(msg)

    def _create_model(self) -> Model:
        from openvino.model_api.adapters import OpenvinoAdapter, create_core, get_user_config
        from openvino.model_api.models import AnomalyDetection

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_adapter = OpenvinoAdapter(
            create_core(),
            self.model_name,
            max_num_requests=self.num_requests,
            plugin_config=plugin_config,
        )

        # Load threshold from the model xml
        self.image_threshold = F1AdaptiveThreshold(
            float(model_adapter.model.get_rt_info(["model_info", "image_threshold"]).value),
        )
        self.pixel_threshold = F1AdaptiveThreshold(
            float(model_adapter.model.get_rt_info(["model_info", "pixel_threshold"]).value),
        )

        return AnomalyDetection.create_model(
            model=model_adapter,
            model_type=self.model_type,
            configuration=self.model_api_configuration,
        )

    def _update_metrics(self, outputs: list[AnomalyResult]) -> None:
        """Update metrics."""
        # [TODO](ashwinvaidya17): Since torchmetrics are used. Need to update this based on the discussion.
        # collate the results
        results = {
            "pred_score": torch.tensor([result.pred_score for result in outputs]),
            "pred_label": torch.tensor([0 if result.pred_label == "Normal" else 1 for result in outputs]),
            # converting to tensor directly from a list is slow
            "pred_mask": torch.tensor(np.array([result.pred_mask for result in outputs]))
            if outputs[0].pred_mask is not None
            else None,
            "anomaly_map": torch.tensor(np.array([result.anomaly_map for result in outputs]))
            if outputs[0].anomaly_map is not None
            else None,
        }
        # Normalize the anomaly map
        results["anomaly_map"] = results["anomaly_map"].long() / 255.0 if results["anomaly_map"] is not None else None
        # compute metrics
        self.image_metrics.update(results["pred_score"], results["pred_label"])
        if results["pred_mask"] is not None and results["anomaly_map"] is not None:
            self.pixel_metrics.update(results["anomaly_map"], results["pred_mask"])

    def configure_callbacks(self) -> Callback:
        """Setup up metrics."""
        image_metrics = ["AUROC", "F1Score"]
        pixel_metrics = image_metrics if self.task != AnomalibTaskType.CLASSIFICATION else None

        self.image_metrics = create_metric_collection(image_metrics, "image_")
        self.pixel_metrics = create_metric_collection(pixel_metrics if pixel_metrics else [], "pixel_")

    def on_test_epoch_start(self) -> None:
        """Reset metrics."""
        self.image_metrics.reset()
        self.pixel_metrics.reset()

    def test_step(self, inputs: AnomalyModelInputs, batch_idx: int) -> list[AnomalyResult]:
        """Return outputs from the OpenVINO model."""
        return self.forward(inputs)  # type: ignore[return-value]

    def on_test_batch_end(
        self,
        outputs: list[AnomalyResult],
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update metrics."""
        self._update_metrics(outputs)

    def on_test_epoch_end(self) -> None:
        """Log metrics."""
        self._log_metrics()

    def on_predict_epoch_start(self) -> None:
        """Reset metrics."""
        self.image_metrics.reset()
        self.pixel_metrics.reset()

    def predict_step(self, inputs: AnomalyModelInputs, batch_idx: int) -> list[AnomalyResult]:
        """Return outputs from the OpenVINO model."""
        return self.forward(inputs)  # type: ignore[return-value]

    def on_predict_batch_end(
        self,
        outputs: list[AnomalyResult],
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update metrics."""
        self._update_metrics(outputs)

    def on_predict_epoch_end(self) -> None:
        """Log metrics."""
        return self._log_metrics()

    def _customize_outputs(self, outputs: list[AnomalyResult], inputs: AnomalyModelInputs) -> list[AnomalyResult]:
        """Return outputs from the OpenVINO model as is."""
        return outputs

    def _log_metrics(self) -> None:
        """Log metrics."""
        self.log_dict(self.image_metrics.compute())
        if self.pixel_metrics._update_called:  # noqa: SLF001
            self.log_dict(self.pixel_metrics.compute())
