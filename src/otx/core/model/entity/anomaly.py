"""Base Anomaly OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnx
import openvino
import torch
from torch import nn

from otx.core.exporter.base import OTXModelExporter
from otx.core.model.entity.base import OTXModel
from otx.core.types.precision import OTXPrecisionType


class _AnomalibLightningArgsCache:
    """Caches args for the anomalib lightning module.

    This is needed as the arguments are passed to the OTX model. These are saved and used by the OTX anomaly
    lightning model.
    """

    def __init__(self):
        self._args: dict[str, Any] = {}

    def update(self, **kwargs) -> None:
        """Add args to cache."""
        self._args.update(kwargs)

    def get(self) -> dict[str, Any]:
        """Get cached args."""
        return self._args


class _AnomalyModelExporter(OTXModelExporter):
    def __init__(
        self,
        transforms: nn.Sequential,
        min_val: float,
        max_val: float,
        image_threshold: float = 0.5,
        pixel_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.transforms: dict[str, int | list[float]] = self._convert_transforms_to_dict(transforms)
        self.pixel_threshold: float = pixel_threshold
        self.image_threshold: float = image_threshold
        self.min_val: float = min_val
        self.max_val: float = max_val

    def _convert_transforms_to_dict(self, transforms: nn.Sequential) -> dict[str, int | list[float]]:
        """Converts transforms to a dictionary."""
        transform_dict = {}
        for transform in transforms:
            name = transform.__class__.__name__
            # Need to revisit this. It is redundant with image_shape
            if "Resize" in name:
                transform_dict["orig_height"] = transform.size
                transform_dict["orig_width"] = transform.size
            elif "Normalize" in name:
                # should be float and in range [0-255]
                transform_dict["mean_values"] = transform.mean
                transform_dict["std_values"] = transform.std
        return transform_dict

    def _get_onnx_metadata(self) -> dict[str, float]:
        """Get metadata from the anomalib model."""
        return {
            "image_threshold": self.image_threshold,
            "pixel_threshold": self.pixel_threshold,
            "min": self.min_val,
            "max": self.max_val,
        }

    def _get_openvino_metadata(self) -> dict[tuple[str, str], float | list[float]] | str:
        onnx_metadata = self._get_onnx_metadata()
        metadata = {
            ("model_info", "image_threshold"): onnx_metadata["image_threshold"],
            ("model_info", "pixel_threshold"): onnx_metadata["pixel_threshold"],
            ("model_info", "normalization_scale"): onnx_metadata["max"] - onnx_metadata["min"],
            ("model_info", "reverse_input_channels"): True,  # convert BGR to RGB in modelAPI
            ("model_info", "model_type"): "AnomalyDetection",
            ("model_info", "labels"): "Normal Anomaly",
            ("model_info", "image_shape"): [self.transforms["orig_height"], self.transforms["orig_width"]],
            ("model_info", "task"): "classification",  # TODO(ashwinvaiday17): Make this dynamic
        }
        # TODO add transform metadata
        for key, value in self.transforms.items():
            metadata[("model_info", key)] = value
        return metadata

    def to_openvino(
        self,
        model: nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        height, width = self.transforms["orig_height"], self.transforms["orig_width"]
        save_path = str(output_dir / f"{base_model_name}.xml")
        metadata = self._get_openvino_metadata()
        exported_model = openvino.convert_model(
            input_model=model,
            example_input=torch.rand(1, 3, height, width).to(next(model.parameters()).device),
        )
        exported_model = _AnomalyModelExporter._embed_openvino_ir_metadata(exported_model, metadata)
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
        height, width = self.transforms["orig_height"], self.transforms["orig_width"]
        save_path = str(output_dir / f"{base_model_name}.onnx")
        torch.onnx.export(
            model=model,
            args=(torch.rand(1, 3, height, width)).to(next(model.parameters()).device),
            f=save_path,
            opset_version=14,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            input_names=["input"],
            output_names=["output"],
        )
        onnx_model = onnx.load(save_path)
        if embed_metadata:
            metadata = self._get_onnx_metadata()
            onnx_model = _AnomalyModelExporter._embed_onnx_metadata(onnx_model, metadata)
        if precision == OTXPrecisionType.FP16:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model, save_path)
        return Path(save_path)


class OTXAnomalyModel(OTXModel):
    """Base Anomaly OTX Model."""

    def __init__(self) -> None:
        self.model: nn.Module
        super().__init__(num_classes=2)
        # This cache is used to get params from the OTX model and pass it into Anomalib Lightning module
        self.anomalib_lightning_args = _AnomalibLightningArgsCache()
        self._transforms = None
        self._image_threshold = None
        self._pixel_threshold = None
        self._min = None
        self._max = None

    @property
    def transforms(self) -> nn.Sequential:
        """Get the transforms."""
        if self._transforms:
            return self._transforms
        msg = "Transforms are not set. Ensure that the model is trained or that the weights are loaded correctly."
        raise ValueError(msg)

    @transforms.setter
    def transforms(self, transforms: list[transforms]) -> None:
        """Set the transforms."""
        self._transforms = nn.Sequential(*transforms)

    @property
    def image_threshold(self) -> float:
        if self._image_threshold:
            return self._image_threshold
        msg = "Image threshold is not set. Ensure that the model is trained or that the weights are loaded correctly."
        raise ValueError(msg)

    @image_threshold.setter
    def image_threshold(self, image_threshold: float) -> None:
        self._image_threshold = image_threshold

    @property
    def pixel_threshold(self) -> float:
        if self._pixel_threshold:
            return self._pixel_threshold
        msg = "Pixel threshold is not set. Ensure that the model is trained or that the weights are loaded correctly."
        raise ValueError(msg)

    @pixel_threshold.setter
    def pixel_threshold(self, pixel_threshold: float) -> None:
        self._pixel_threshold = pixel_threshold

    @property
    def min_val(self) -> float:
        if self._min:
            return self._min
        msg = "Min is not set. Ensure that the model is trained or that the weights are loaded correctly."
        raise ValueError(msg)

    @min_val.setter
    def min_val(self, min_val: float) -> None:
        self._min = min_val

    @property
    def max_val(self) -> float:
        if self._max:
            return self._max
        msg = "Max is not set. Ensure that the model is trained or that the weights are loaded correctly."
        raise ValueError(msg)

    @max_val.setter
    def max_val(self, max_val: float) -> None:
        self._max = max_val

    def _customize_inputs(self, *_, **__) -> None:
        """Input customization is done through the lightning module."""

    def _customize_outputs(self, *_, **__) -> None:
        """Output customization is done through the lightning module."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Call forward on the raw tensor.

        Overrides the base forward as input and output customization occurs from the lightning model.
        """
        return self.model(input_tensor)

    @property
    def _exporter(self) -> OTXModelExporter:
        """Get the model exporter."""
        return _AnomalyModelExporter(
            transforms=self.transforms,
            min_val=self.min_val,
            max_val=self.max_val,
            image_threshold=self.image_threshold,
            pixel_threshold=self.pixel_threshold,
        )
