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
    ) -> Path:
        """Export to ONNX format.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

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
