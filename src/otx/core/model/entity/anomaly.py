"""Base Anomaly OTX model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import onnx
import openvino
import torch
from anomalib import TaskType
from torch import nn

from otx.core.data.dataset.base import LabelInfo
from otx.core.data.entity.anomaly import (
    AnomalyClassificationBatchPrediction,
    AnomalyDetectionBatchPrediction,
    AnomalySegmentationBatchPrediction,
)
from otx.core.data.entity.base import T_OTXBatchDataEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.model.entity.base import OTXModel, OVModel
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from openvino.model_api.models import Model
    from openvino.model_api.models.anomaly import AnomalyResult
    from torchvision.transforms.v2 import Transform


class _AnomalibLightningArgsCache:
    """Caches args for the anomalib lightning module.

    This is needed as the arguments are passed to the OTX model. These are saved and used by the OTX anomaly
    lightning model.
    """

    def __init__(self) -> None:
        self._args: dict[str, Any] = {}

    def update(self, **kwargs) -> None:
        """Add args to cache."""
        self._args.update(kwargs)

    def get(self) -> dict[str, Any]:
        """Get cached args."""
        return self._args


@dataclass
class _OVModelInfo:
    """OpenVINO model information."""

    image_shape: tuple[int, int] = (256, 256)
    image_threshold: float = 0.5
    pixel_threshold: float = 0.5
    task: TaskType = TaskType.CLASSIFICATION
    # the actual values for mean and scale should be in range 0-255
    mean_values: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_values: tuple[float, float, float] = (1.0, 1.0, 1.0)
    normalization_scale: float = 1.0
    orig_height: int = 256
    orig_width: int = 256


class _AnomalyModelExporter(OTXModelExporter):
    def __init__(
        self,
        model_info: _OVModelInfo,
    ) -> None:
        self.model_info = model_info
        metadata = {
            ("model_info", "image_threshold"): model_info.image_threshold,
            ("model_info", "pixel_threshold"): model_info.pixel_threshold,
            ("model_info", "normalization_scale"): model_info.normalization_scale,
            ("model_info", "orig_height"): model_info.orig_height,
            ("model_info", "orig_width"): model_info.orig_width,
            ("model_info", "image_shape"): model_info.image_shape,
            ("model_info", "labels"): "Normal Anomaly",
            ("model_info", "model_type"): "AnomalyDetection",
            ("model_info", "task"): model_info.task.value,
        }
        super().__init__(
            input_size=(1, 3, *model_info.image_shape),
            mean=model_info.mean_values,
            std=model_info.scale_values,
            swap_rgb=False,  # default value. Ideally, modelAPI should pass RGB inputs after the pre-processing step
            metadata=metadata,
        )

    def to_openvino(
        self,
        model: nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
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
        save_path = str(output_dir / f"{base_model_name}.onnx")
        torch.onnx.export(
            model=model,
            args=(torch.rand(1, 3, self.model_info.orig_height, self.model_info.orig_width)).to(
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


class OTXAnomalyModel(OTXModel):
    """Base Anomaly OTX Model."""

    def __init__(self) -> None:
        self.model: nn.Module
        super().__init__(num_classes=2)
        # This cache is used to get params from the OTX model and pass it into Anomalib Lightning module
        self.anomalib_lightning_args = _AnomalibLightningArgsCache()
        self.model_info = _OVModelInfo()

    def extract_model_info_from_transforms(self, transforms: list[Transform]) -> None:
        """Stores values from transforms to ``self.model_info``."""
        for transform in transforms:
            name = transform.__class__.__name__
            # Need to revisit this. It is redundant with image_shape
            if "Resize" in name:
                self.model_info.orig_height = transform.size
                self.model_info.orig_width = transform.size
                self.model_info.image_shape = (transform.size, transform.size)
            elif "Normalize" in name:
                # should be float and in range [0-255]
                self.model_info.mean_values = transform.mean
                self.model_info.scale_values = transform.std

    @property
    def _exporter(self) -> OTXModelExporter:
        """Get the model exporter."""
        return _AnomalyModelExporter(
            model_info=self.model_info,
        )

    @property
    def task_type(self) -> TaskType:
        """Return task type in Anomalib's format."""
        if self._task_type:
            return self._task_type
        msg = "Task type is not set."
        raise ValueError(msg)

    @task_type.setter
    def task_type(self, task_type: TaskType) -> None:
        self._task_type = task_type

    @property
    def label_info(self) -> LabelInfo:
        """Get this model label information."""
        return self._label_info

    @label_info.setter
    def label_info(self, value: LabelInfo | list[str]) -> None:
        """Set this model label information.

        It changes the number of classes to 2 and sets the labels as Normal and Anomaly.
        This is because Datumaro returns multiple classes from the dataset. If self.label_info != 2,
        then It will call self._reset_prediction_layer() to reset the prediction layer. Which is not required.
        """
        if isinstance(value, list):
            # value can be greater than 2 as datumaro returns all anomalous categories separately
            self._label_info = LabelInfo(label_names=["Normal", "Anomaly"], label_groups=[["Normal", "Anomaly"]])
        else:
            self._label_info = value

    @property
    def trainable_model(self) -> str | None:
        """Use this to return the name of the model that needs to be trained.

        This might not be the cleanest solution.

        Some models have multiple architectures and only one of them needs to be trained.
        However the optimizer is configured in the Anomalib's lightning model. This can be used
        to inform the OTX lightning model which model to train.
        """
        return None

    def _customize_inputs(self, inputs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Input customization is done through the lightning module."""
        return inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: T_OTXBatchDataEntity,
    ) -> None:
        """Output customization is done through the lightning module."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Call forward on the raw tensor.

        Overrides the base forward as input and output customization occurs from the lightning model.
        """
        return self.model(input_tensor)


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
