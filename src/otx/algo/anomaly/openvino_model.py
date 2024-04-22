"""OTX Anomaly OpenVINO model.

All anomaly models use the same AnomalyDetection model from ModelAPI.
"""

# TODO(someone): Revisit mypy errors after OTXLitModule deprecation and anomaly refactoring
# mypy: ignore-errors

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.core.metrics.types import MetricCallable, NullMetricCallable
from otx.core.model.anomaly import AnomalyModelInputs
from otx.core.model.base import OVModel

if TYPE_CHECKING:
    from model_api.models import Model
    from model_api.models.anomaly import AnomalyResult


class AnomalyOpenVINO(OVModel):
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
        metric: MetricCallable = NullMetricCallable,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type="AnomalyDetection",
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _create_model(self) -> Model:
        from model_api.adapters import OpenvinoAdapter, create_core, get_user_config
        from model_api.models import AnomalyDetection

        plugin_config = get_user_config("AUTO", str(self.num_requests), "AUTO")
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_adapter = OpenvinoAdapter(
            create_core(),
            self.model_name,
            max_num_requests=self.num_requests,
            plugin_config=plugin_config,
        )
        return AnomalyDetection.create_model(
            model=model_adapter,
            model_type=self.model_type,
            configuration=self.model_api_configuration,
        )

    def test_step(self, inputs: AnomalyModelInputs, batch_idx: int) -> list[AnomalyResult]:
        """Return outputs from the OpenVINO model."""
        return self.forward(inputs)  # type: ignore[return-value]

    def predict_step(self, inputs: AnomalyModelInputs, batch_idx: int) -> list[AnomalyResult]:
        """Return outputs from the OpenVINO model."""
        return self.forward(inputs)  # type: ignore[return-value]

    def _customize_outputs(self, outputs: list[AnomalyResult], inputs: AnomalyModelInputs) -> list[AnomalyResult]:
        """Return outputs from the OpenVINO model as is."""
        return outputs
