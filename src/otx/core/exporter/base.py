# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for base model exporter used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from otx.core.types.export import OTXExportPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    import onnx
    import openvino
    import torch


class OTXModelExporter:
    """Base class for the model exporters used in OTX."""

    @abstractmethod
    def to_openvino(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights
            metadata (dict[tuple[str, str],str] | None, optional): metadata to embed to the exported model.
        """

    @abstractmethod
    def to_onnx(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], str] | None = None,
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
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> None:
        """Export to zip folder final OV IR model with runable demo.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights
            metadata (dict[tuple[str, str],str] | None, optional): metadata to embed to the exported model.
        """

        from zipfile import ZipFile
        import os
        import io
        import json
        from otx.core.exporter import exportable_code



        work_dir = os.path.dirname(exportable_code.__file__)
        parameters = {}
        parameters["type_of_model"] = ""
        parameters["converter_type"] = ""
        parameters["model_parameters"] = {}
        zip_buffer = io.BytesIO()
        with ZipFile(zip_buffer, "w") as arch:
            # model files
            path_to_model = self.to_openvino(model,
                                            output_dir,
                                            base_model_name,
                                            precision,
                                            metadata)
            arch.writestr(os.path.join("model", "model.xml"), str(path_to_model))
            arch.writestr(os.path.join("model", "model.bin"), str(path_to_model)[:-4] + ".bin")

            arch.writestr(
                os.path.join("model", "config.json"),
                json.dumps(parameters, ensure_ascii=False, indent=4),
            )
            # python files
            arch.write(
                os.path.join(work_dir, "requirements.txt"),
                os.path.join("python", "requirements.txt"),
            )
            arch.write(os.path.join(work_dir, "LICENSE"), os.path.join("python", "LICENSE"))
            arch.write(os.path.join(work_dir, "demo.py"), os.path.join("python", "demo.py"))
            arch.write(os.path.join(work_dir, "README.md"), os.path.join(".", "README.md"))
        # output_model.exportable_code = zip_buffer.getvalue()
        with open(output_dir / "exportable_code.zip", 'wb') as f:
            f.write(zip_buffer.getvalue())

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
