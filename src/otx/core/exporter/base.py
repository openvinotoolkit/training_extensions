# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base model exporter used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    import onnx
    import openvino
    import torch


class OTXModelExporter:
    """Base class for the model exporters used in OTX.

    Args:
        input_size (tuple[int, ...]): Input shape.
        mean (tuple[float, float, float], optional): Mean values of 3 channels. Defaults to (0.0, 0.0, 0.0).
        std (tuple[float, float, float], optional): Std values of 3 channels. Defaults to (1.0, 1.0, 1.0).
        resize_mode (Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"], optional):
            A resize type for model preprocess. "standard" resizes iamges without keeping ratio.
            "fit_to_window" resizes images while keeping ratio.
            "fit_to_window_letterbox" resizes images and pads images to fit the size. Defaults to "standard".
        pad_value (int, optional): Padding value. Defaults to 0.
        swap_rgb (bool, optional): Whether to convert the image from BGR to RGB Defaults to False.
        metadata (dict[tuple[str, str],str] | None, optional): metadata to embed to the exported model.
    """

    def __init__(
        self,
        input_size: tuple[int, ...],
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        resize_mode: Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"] = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.input_size = input_size
        self.mean = mean
        self.std = std
        self.resize_mode = resize_mode
        self.pad_value = pad_value
        self.swap_rgb = swap_rgb
        self.metadata = metadata

    def export(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        export_format: OTXExportFormatType = OTXExportFormatType.OPENVINO,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Exports input model to the specified deployable format, such as OpenVINO IR or ONNX.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            format (OTXExportFormatType): final format of the exported model
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

        Returns:
            Path: path to the exported model
        """
        if export_format == OTXExportFormatType.OPENVINO:
            return self.to_openvino(model, output_dir, base_model_name, precision)
        if export_format == OTXExportFormatType.ONNX:
            return self.to_onnx(model, output_dir, base_model_name, precision)

        msg = f"Unsupported export format: {export_format}"
        raise ValueError(msg)

    @abstractmethod
    def to_openvino(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

        Returns:
            Path: path to the exported model.
        """

    @abstractmethod
    def to_onnx(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Export to ONNX format.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights
            embed_metadata (bool): flag which enables embedding of metadata to the ONNX model.
            Metadata embedding should be enabled if model is going to be converted to OV IR
            (otherwise OV fails on the resulting model).

        Returns:
            Path: path to the exported model.
        """

    @staticmethod
    def _embed_onnx_metadata(onnx_model: onnx.ModelProto, metadata: dict[tuple[str, str], Any]) -> onnx.ModelProto:
        """Embeds metadata to ONNX model."""
        for item in metadata:
            meta = onnx_model.metadata_props.add()
            attr_path = " ".join(map(str, item))
            meta.key = attr_path.strip()
            meta.value = str(metadata[item])

        return onnx_model

    @staticmethod
    def _embed_openvino_ir_metadata(ov_model: openvino.Model, metadata: dict[tuple[str, str], Any]) -> openvino.Model:
        """Embeds metadata to OpenVINO model."""
        for k, data in metadata.items():
            ov_model.set_rt_info(data, list(k))

        return ov_model

    def _extend_model_metadata(self, metadata: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
        """Extends metadata coming from model with preprocessing-specific parameters.

        Model's original metadata has priority over exporter's extra metadata

        Args:
            metadata (dict[tuple[str, str], str]): _description_

        Returns:
            dict[tuple[str, str] ,str]: updated metadata
        """
        mean_str = " ".join(map(str, self.mean))
        std_str = " ".join(map(str, self.std))

        extra_data = {
            ("model_info", "mean_values"): mean_str.strip(),
            ("model_info", "scale_values"): std_str.strip(),
            ("model_info", "resize_type"): self.resize_mode,
            ("model_info", "pad_value"): str(self.pad_value),
            ("model_info", "reverse_input_channels"): str(self.swap_rgb),
        }
        extra_data.update(metadata)

        return extra_data

    def _postprocess_openvino_model(self, exported_model: openvino.Model) -> openvino.Model:
        # workaround for OVC's bug: single output doesn't have a name in OV model
        if len(exported_model.outputs) == 1 and len(exported_model.outputs[0].get_names()) == 0:
            exported_model.outputs[0].tensor.set_names({"output1"})

        if self.metadata is not None:
            export_metadata = self._extend_model_metadata(self.metadata)
            exported_model = self._embed_openvino_ir_metadata(exported_model, export_metadata)

        return exported_model

    def _postprocess_onnx_model(
        self,
        onnx_model: onnx.ModelProto,
        embed_metadata: bool,
        precision: OTXPrecisionType
    ) -> onnx.ModelProto:
        if embed_metadata:
            metadata = {} if metadata is None else self._extend_model_metadata(self.metadata)
            onnx_model = self._embed_onnx_metadata(onnx_model, metadata)

        onnx_model = self._embed_onnx_metadata(onnx_model, metadata)
        if precision == OTXPrecisionType.FP16:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)

        return onnx_model
