# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base model exporter used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    import onnx
    import openvino
    import torch


class OTXModelExporter:
    """Base class for the model exporters used in OTX."""

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
            metadata (dict[tuple[str, str],str] | None, optional): metadata to embed to the exported model.

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
            metadata (dict[tuple[str, str],str] | None, optional): metadata to embed to the exported model.
        """

    def to_exportable_code(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> Path:
        """Export to zip folder final OV IR model with runable demo.

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
        import io
        import json
        import os
        import tempfile
        from zipfile import ZipFile

        from otx.core.exporter.exportable_code import demo

        work_dir = Path(demo.__file__).parent
        parameters: dict[str, Any] = {}
        if metadata is not None:
            parameters["type_of_model"] = metadata.get(("model_info", "task_type"), "")
            parameters["converter_type"] = metadata.get(("model_info", "model_type"), "")
            parameters["model_parameters"] = {
                "labels": metadata.get(("model_info", "labels"), ""),
                "labels_ids": metadata.get(("model_info", "label_ids"), ""),
            }
        zip_buffer = io.BytesIO()
        temp_dir = tempfile.TemporaryDirectory()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            path_to_model = self.to_openvino(model, Path(temp_dir.name), base_model_name, precision, metadata)
            arch.write(str(path_to_model), Path("model") / "model.xml")
            arch.write(str(path_to_model)[:-4] + ".bin", Path("model") / "model.bin")

            arch.writestr(
                str(Path("model") / "config.json"),
                json.dumps(parameters, ensure_ascii=False, indent=4),
            )
            # python files
            arch.write(
                Path(work_dir) / "requirements.txt",
                Path("python") / "requirements.txt",
            )
            arch.write(Path(work_dir, "LICENSE"), Path("python") / "LICENSE")
            arch.write(Path(work_dir, "demo.py"), Path("python") / "demo.py")
            arch.write(Path(work_dir, "README.md"), Path("./") / "README.md")
            # write demo_package
            demo_package = Path(work_dir, "demo_package")
            for root, _, files in os.walk(demo_package):
                if root.endswith("__pycache__"):
                    continue
                for file in files:
                    file_path = Path(root) / file
                    archive_path = file_path.relative_to(demo_package)
                    arch.write(file_path, Path("python") / "demo_package" / archive_path)
        # save archive
        output_path = output_dir / "exportable_code.zip"
        with Path.open(output_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        temp_dir.cleanup()
        return output_path

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
