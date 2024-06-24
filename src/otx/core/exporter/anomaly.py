# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for anomaly models exporter used in OTX."""
from __future__ import annotations

from pathlib import Path

import onnx
import openvino
import torch
from anomalib import TaskType as AnomalibTaskType
from torch import nn

from otx.core.exporter.base import OTXModelExporter
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import NullLabelInfo
from otx.core.types.precision import OTXPrecisionType


class OTXAnomalyModelExporter(OTXModelExporter):
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
        """
        self.orig_height, self.orig_width = image_shape
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold
        self.task = task
        self.normalization_scale = normalization_scale

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

    def to_openvino(
        self,
        model: nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Exports the model to OpenVINO Intermediate Representation.

        Args:
            model (nn.Module): The model to export.
            output_dir (Path): The directory where the exported model will be saved.
            base_model_name (str, optional): The base name for the exported model. Defaults to "exported_model".
            precision (OTXPrecisionType, optional): The precision type for the exported model.
            Defaults to OTXPrecisionType.FP32.

        Returns:
            Path: The path to the exported model.
        """
        save_path = str(output_dir / f"{base_model_name}.xml")
        exported_model = openvino.convert_model(
            input_model=model,
            example_input=torch.rand(self.input_size),
            input=(openvino.runtime.PartialShape(self.input_size)),
        )
        exported_model = self._postprocess_openvino_model(exported_model)
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXPrecisionType.FP16))
        return Path(save_path)

    def to_onnx(
        self,
        model: nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Exports the model to ONNX format.

        Args:
            model (nn.Module): The model to export.
            output_dir (Path): The directory where the exported model will be saved.
            base_model_name (str, optional): The base name for the exported model. Defaults to "exported_model".
            precision (OTXPrecisionType, optional): The precision type for the exported model.
            Defaults to OTXPrecisionType.FP32.
            embed_metadata (bool, optional): Whether to embed metadata in the exported model. Defaults to True.

        Returns:
            Path: The path to the exported model.
        """
        save_path = str(output_dir / f"{base_model_name}.onnx")
        torch.onnx.export(
            model=model,
            args=(torch.rand(1, 3, self.orig_height, self.orig_width)).to(
                next(model.parameters()).device,
            ),
            f=save_path,
            opset_version=14,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            input_names=["input"],
            output_names=["output"],
        )
        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)
        onnx.save(onnx_model, save_path)
        return Path(save_path)
