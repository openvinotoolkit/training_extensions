# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base model exporter used in OTX."""

from __future__ import annotations

import json
import os
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from zipfile import ZipFile

from otx.core.exporter.exportable_code import demo
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
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
            A resize type for model preprocess. "standard" resizes images without keeping ratio.
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
        export_args: dict[str, Any] | None = None,
    ) -> Path:
        """Exports input model to the specified deployable format, such as OpenVINO IR or ONNX.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            format (OTXExportFormatType): final format of the exported model
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights
            export_args (dict, optional): manual arguments for the export function.
                If not provided, the exporter will set dummy inputs

        Returns:
            Path: path to the exported model
        """
        if export_format == OTXExportFormatType.OPENVINO:
            return self.to_openvino(model, output_dir, base_model_name, precision, export_args)
        if export_format == OTXExportFormatType.ONNX:
            return self.to_onnx(model, output_dir, base_model_name, precision, export_args=export_args)
        if export_format == OTXExportFormatType.EXPORTABLE_CODE:
            return self.to_exportable_code(model, output_dir, base_model_name, precision)

        msg = f"Unsupported export format: {export_format}"
        raise ValueError(msg)

    @abstractmethod
    def to_openvino(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        export_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights
            export_args (dict, optional): manual arguments for the export function.
                If not provided, the exporter will set dummy inputs

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
        export_args: dict[str, Any] | None = None,
    ) -> Path:
        """Abstract method for ONNX export.

        Converts the given torch model to ONNX format and saves it to the specified output directory.

        Args:
            model (torch.nn.Module): The input PyTorch model to be converted.
            output_dir (Path): The directory where the ONNX model will be saved.
            base_model_name (str, optional): The name of the exported ONNX model. Defaults to "exported_model".
            precision (OTXPrecisionType, optional): The precision type for the exported model.
            Defaults to OTXPrecisionType.FP32.
            embed_metadata (bool, optional): Flag to embed metadata in the exported ONNX model. Defaults to True.
            export_args (dict, optional): manual arguments for the export function.
                If not provided, the exporter will set dummy inputs

        Returns:
            Path: The file path where the ONNX model is saved.
        """

    def to_exportable_code(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export to zip folder final OV IR model with runable demo.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

        Returns:
            Path: path to the exported model.
        """
        work_dir = Path(demo.__file__).parent
        parameters: dict[str, Any] = {}
        if self.metadata is not None:
            parameters["type_of_model"] = self.metadata.get(("model_info", "task_type"), "")
            parameters["converter_type"] = self.metadata.get(("model_info", "model_type"), "")
            parameters["model_parameters"] = {
                "labels": self.metadata.get(("model_info", "labels"), ""),
                "labels_ids": self.metadata.get(("model_info", "label_ids"), ""),
            }

        output_zip_path = output_dir / "exportable_code.zip"
        Path.mkdir(output_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir, ZipFile(output_zip_path, "x") as arch:
            # model files
            path_to_model = self.to_openvino(model, Path(temp_dir), base_model_name, precision)
            arch.write(str(path_to_model), Path("model") / "model.xml")
            arch.write(path_to_model.with_suffix(".bin"), Path("model") / "model.bin")

            arch.writestr(
                str(Path("model") / "config.json"),
                json.dumps(parameters, ensure_ascii=False, indent=4),
            )
            # python files
            arch.write(
                work_dir / "requirements.txt",
                Path("python") / "requirements.txt",
            )
            arch.write(work_dir.parents[5] / "LICENSE", Path("python") / "LICENSE")
            arch.write(work_dir / "demo.py", Path("python") / "demo.py")
            arch.write(work_dir / "README.md", Path("./") / "README.md")
            arch.write(work_dir / "setup.py", Path("python") / "setup.py")
            # write demo_package
            demo_package = work_dir / "demo_package"
            for root, _, files in os.walk(demo_package):
                if root.endswith("__pycache__"):
                    continue
                for file in files:
                    file_path = Path(root) / file
                    archive_path = file_path.relative_to(demo_package)
                    arch.write(file_path, Path("python") / "demo_package" / archive_path)
        return output_zip_path

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
            metadata (dict[tuple[str, str], str]): existing metadata for export

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
        precision: OTXPrecisionType,
    ) -> onnx.ModelProto:
        if embed_metadata:
            metadata = {} if self.metadata is None else self._extend_model_metadata(self.metadata)
            onnx_model = self._embed_onnx_metadata(onnx_model, metadata)

        if precision == OTXPrecisionType.FP16:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)

        return onnx_model
