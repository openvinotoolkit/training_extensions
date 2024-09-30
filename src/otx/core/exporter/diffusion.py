# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Exporter for diffusion models that uses native torch and OpenVINO conversion tools."""

from __future__ import annotations

import logging as log
from pathlib import Path

import onnx
import openvino
import torch

from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.model.base import OTXModel
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType


class DiffusionOTXModelExporter(OTXNativeModelExporter):
    """Exporter for diffusion models that uses native torch and OpenVINO conversion tools."""

    def export(  # type: ignore[override]
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        export_format: OTXExportFormatType = OTXExportFormatType.OPENVINO,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        to_exportable_code: bool = False,
    ) -> Path:
        """Exports input model to the specified deployable format, such as OpenVINO IR or ONNX.

        Args:
            model (OTXModel): OTXModel to be exported
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            format (OTXExportFormatType): final format of the exported model
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights
            to_exportable_code (bool, optional): whether to generate exportable code.
                Currently not supported by Diffusion task.

        Returns:
            Path: path to the exported model
        """
        if export_format == OTXExportFormatType.OPENVINO:
            if to_exportable_code:
                msg = "Exportable code option is not supported and will be ignored."
                log.warning(msg)
            fn = self.to_openvino
        else:
            fn = self.to_onnx  # type: ignore[assignment]

        return fn(model, output_dir, base_model_name, precision)

    def to_openvino(
        self,
        model: OTXModel | torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        In this implementation the export is done only via standard OV/ONNX tools.
        """
        exported_model = openvino.convert_model(
            model,
            example_input=self.onnx_export_configuration["args"],
            input={k: v.shape for k, v in self.onnx_export_configuration["args"].items()},
        )
        exported_model = self._postprocess_openvino_model(exported_model)

        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXPrecisionType.FP16))
        log.info("Converting to OpenVINO is done.")

        return Path(save_path)

    def to_onnx(
        self,
        model: OTXModel | torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
        model_type: str = "stable_diffusion",
    ) -> Path:
        """Export the given PyTorch model to ONNX format and save it to the specified output directory.

        Args:
            model (OTXModel): OTXModel to be exported.
            output_dir (Path): The directory where the ONNX model will be saved.
            base_model_name (str, optional): The base name for the exported model. Defaults to "exported_model".
            precision (OTXPrecisionType, optional): The precision type for the exported model.
            Defaults to OTXPrecisionType.FP32.
            embed_metadata (bool, optional): Whether to embed metadata in the ONNX model. Defaults to True.

        Returns:
            Path: The path to the saved ONNX model.
        """
        save_path = str(output_dir / (base_model_name + ".onnx"))

        torch.onnx.export(
            model=model,
            f=save_path,
            **self.onnx_export_configuration,
        )

        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, False, precision)

        if self.metadata is not None and embed_metadata:
            export_metadata = self._extend_model_metadata(self.metadata)
            export_metadata[("model_info", "model_type")] = model_type
            onnx_model = self._embed_onnx_metadata(onnx_model, export_metadata)

        onnx.save(onnx_model, save_path)
        log.info("Converting to ONNX is done.")

        return Path(save_path)
