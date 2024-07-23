# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for anomaly models exporter used in OTX."""
from __future__ import annotations

from typing import Any

from anomalib import TaskType as AnomalibTaskType

from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import NullLabelInfo


class OTXAnomalyModelExporter(OTXNativeModelExporter):
    """Exporter for anomaly tasks."""

    def __init__(
        self,
        image_shape: tuple[int, int] = (256, 256),
        image_threshold: float = 0.5,
        pixel_threshold: float = 0.5,
        task: AnomalibTaskType = AnomalibTaskType.CLASSIFICATION,
        # the actual values for mean and scale should be in range 0-255
        mean_values: tuple[float, float, float] = (0.0, 0.0, 0.0),
        scale_values: tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalization_scale: float = 1.0,
        via_onnx: bool = False,
        onnx_export_configuration: dict[str, Any] | None = None,
    ) -> None:
        """Initializes `OTXAnomalyModelExporter` object.

        Args:
            image_shape (tuple[int, int], optional): Shape of the input image.
                Defaults to (256, 256).
            image_threshold (float, optional): Threshold for image anomaly detection.
                Defaults to 0.5.
            pixel_threshold (float, optional): Threshold for pixel anomaly detection.
                Defaults to 0.5.
            task (AnomalibTaskType, optional): Task type for anomaly detection.
                Defaults to AnomalibTaskType.CLASSIFICATION.
            mean_values (tuple[float, float, float], optional): Mean values for normalization.
                Defaults to (0.0, 0.0, 0.0).
            scale_values (tuple[float, float, float], optional): Scale values for normalization.
                Defaults to (1.0, 1.0, 1.0).
            normalization_scale (float, optional): Scale value for normalization.
                Defaults to 1.0.
            via_onnx (bool, optional): Whether to export the model in OpenVINO format via ONNX first.
                Defaults to False.
            onnx_export_configuration (dict[str, Any] | None, optional): Configuration for ONNX export.
                Defaults to None.
        """
        self.orig_height, self.orig_width = image_shape
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold
        self.task = task
        self.normalization_scale = normalization_scale
        self.via_onnx = via_onnx
        self.onnx_export_configuration = onnx_export_configuration if onnx_export_configuration is not None else {}

        super().__init__(
            task_level_export_parameters=TaskLevelExportParameters(
                model_type="anomaly",
                task_type="anomaly",
                label_info=NullLabelInfo(),
                optimization_config={},
            ),
            input_size=(1, 3, *image_shape),
            mean=mean_values,
            std=scale_values,
            swap_rgb=False,
        )

    @property
    def metadata(self) -> dict[tuple[str, str], str | float | int | tuple[int, int]]:  # type: ignore[override]
        """Returns a dictionary containing metadata about the model.

        Returns:
            dict[tuple[str, str], str | float | int | tuple[int, int]]: A dictionary with metadata.
        """
        return {
            ("model_info", "image_threshold"): self.image_threshold,
            ("model_info", "pixel_threshold"): self.pixel_threshold,
            ("model_info", "normalization_scale"): self.normalization_scale,
            ("model_info", "orig_height"): self.orig_height,
            ("model_info", "orig_width"): self.orig_width,
            ("model_info", "image_shape"): (self.orig_height, self.orig_width),
            ("model_info", "labels"): "Normal Anomaly",
            ("model_info", "model_type"): "AnomalyDetection",
            ("model_info", "task"): self.task.value,
        }
