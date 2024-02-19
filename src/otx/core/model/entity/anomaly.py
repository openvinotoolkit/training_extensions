"""Base Anomaly OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from otx.core.data.entity.anomaly import (
    AnomalyClassificationBatchPrediction,
    AnomalyDetectionBatchPrediction,
    AnomalySegmentationBatchPrediction,
)
from otx.core.model.entity.base import OVModel

if TYPE_CHECKING:
    from openvino.model_api.models import Model
    from openvino.model_api.models.anomaly import AnomalyResult


class OVAnomalyModel(OVModel):
    """OTXModel that contains modelAPI's AnomalyModel as its model.

    This uses the inferencer from modelAPI to generate result.
    """

    def __init__(
        self,
        model_name: str,
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        num_classes: int = 2,  # Unused as the model is always 2 classes but needed for kwargs
    ) -> None:
        super().__init__(
            num_classes=2,
            model_name=model_name,
            model_type="AnomalyDetection",
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
        )

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
        return AnomalyDetection.create_model(
            model=model_adapter,
            model_type=self.model_type,
            configuration=self.model_api_configuration,
        )

    def _customize_outputs(
        self,
        outputs: list[AnomalyResult],
        inputs: AnomalyClassificationBatchPrediction
        | AnomalyDetectionBatchPrediction
        | AnomalySegmentationBatchPrediction,
    ) -> list[AnomalyResult]:
        return outputs
