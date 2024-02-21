# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for native model exporter used in OTX."""

from __future__ import annotations

import logging as log
import tempfile
from pathlib import Path
from typing import Any, Literal

import onnx
import openvino
import torch

from otx.core.exporter.base import OTXModelExporter
from otx.core.types.precision import OTXPrecisionType


class OTXNativeModelExporter(OTXModelExporter):
    """Exporter that uses native torch and OpenVINO conversion tools."""

    def __init__(
        self,
        input_size: tuple[int, ...],
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        resize_mode: Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"] = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        metadata: dict[tuple[str, str], str] | None = None,
        via_onnx: bool = False,
        onnx_export_configuration: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(input_size, mean, std, resize_mode, pad_value, swap_rgb, metadata)
        self.via_onnx = via_onnx
        self.onnx_export_configuration = onnx_export_configuration if onnx_export_configuration is not None else {}

    def to_openvino(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        export_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        In this implementation the export is done only via standard OV/ONNX tools.
        """
        if self.via_onnx:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_dir = Path(tmpdirname)

                self.to_onnx(
                    model,
                    tmp_dir,
                    base_model_name,
                    OTXPrecisionType.FP32,
                    False,
                    export_args,
                )
                exported_model = openvino.convert_model(
                    tmp_dir / (base_model_name + ".onnx"),
                    input=tuple(x.shape for x in export_args["args"]),
                )
        else:
            if export_args is None:
                export_args = {
                    "input": (openvino.runtime.PartialShape(self.input_size),),
                    "example_input": torch.rand(self.input_size).to(next(model.parameters()).device)
                }
            export_args.update({"input_model": model})
            exported_model = openvino.convert_model(**export_args)
        exported_model = self._postprocess_openvino_model(exported_model)

        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXPrecisionType.FP16))
        log.info("Converting to OpenVINO is done.")

        return Path(save_path)

    def to_onnx(
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
        export_args: dict[str, Any] | None = None,
    ) -> Path:
        """Export the given PyTorch model to ONNX format and save it to the specified output directory.

        Args:
            model (torch.nn.Module): The PyTorch model to be exported.
            output_dir (Path): The directory where the ONNX model will be saved.
            base_model_name (str, optional): The base name for the exported model. Defaults to "exported_model".
            precision (OTXPrecisionType, optional): The precision type for the exported model.
            Defaults to OTXPrecisionType.FP32.
            embed_metadata (bool, optional): Whether to embed metadata in the ONNX model. Defaults to True.
            export_args (dict, optional): Manual arguments for the export function. If not provided, the exporter will set dummy inputs.

        Returns:
            Path: The path to the saved ONNX model.
        """
        if export_args is None:
            export_args = {"args": torch.rand(self.input_size).to(next(model.parameters()).device)}

        save_path = str(output_dir / (base_model_name + ".onnx"))
        export_args.update({"model": model, "f": save_path})

        torch.onnx.export(**export_args, **self.onnx_export_configuration)

        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        onnx.save(onnx_model, save_path)
        log.info("Converting to ONNX is done.")

        return Path(save_path)
