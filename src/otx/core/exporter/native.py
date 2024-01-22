# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for native model exporter used in OTX."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import onnx
import openvino
import torch

from otx.core.exporter.base import OTXModelExporter
from otx.core.types.export import OTXExportPrecisionType


class OTXNativeModelExporter(OTXModelExporter):
    """Exporter that uses native torch and OpenVINO conversion tools."""

    def __init__(
        self,
        input_size: tuple[int, ...],
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        resize_mode: str = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        via_onnx: bool = False,
        onnx_export_configuration: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(input_size, mean, std, resize_mode, pad_value, swap_rgb)
        self.via_onnx = via_onnx
        self.onnx_export_configuration = onnx_export_configuration if onnx_export_configuration is not None else {}

    def to_openvino(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        In this implementation the export is done only via standard OV/ONNX tools.
        """
        dummy_tensor = torch.rand(self.input_size).to(next(model.parameters()).device)

        if self.via_onnx:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_dir = Path(tmpdirname)

                self.to_onnx(
                    model,
                    tmp_dir,
                    base_model_name,
                    OTXExportPrecisionType.FP32,
                    None,
                )
                exported_model = openvino.convert_model(
                    tmp_dir / (base_model_name + ".onnx"),
                    input=(openvino.runtime.PartialShape(self.input_size),),
                )
        else:
            exported_model = openvino.convert_model(
                model,
                example_input=dummy_tensor,
                input=(openvino.runtime.PartialShape(self.input_size),),
            )

        # workaround for OVC's bug: single output doesn't have a name in OV model
        if len(exported_model.outputs) == 1 and len(exported_model.outputs[0].get_names()) == 0:
            exported_model.outputs[0].tensor.set_names({"output1"})

        metadata = {} if metadata is None else self._extend_model_metadata(metadata)
        exported_model = OTXNativeModelExporter._embed_openvino_ir_metadata(exported_model, metadata)
        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXExportPrecisionType.FP16))

        return Path(save_path)

    def to_onnx(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXExportPrecisionType = OTXExportPrecisionType.FP32,
        metadata: dict[tuple[str, str], str] | None = None,
    ) -> Path:
        """Export to ONNX format.

        In this implementation the export is done only via standard OV/ONNX tools.
        """
        dummy_tensor = torch.rand(self.input_size).to(next(model.parameters()).device)
        save_path = str(output_dir / (base_model_name + ".onnx"))
        metadata = {} if metadata is None else self._extend_model_metadata(metadata)

        torch.onnx.export(model, dummy_tensor, save_path, **self.onnx_export_configuration)

        onnx_model = onnx.load(save_path)
        onnx_model = OTXNativeModelExporter._embed_onnx_metadata(onnx_model, metadata)
        if precision == OTXExportPrecisionType.FP16:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(onnx_model, save_path)

        return Path(save_path)
