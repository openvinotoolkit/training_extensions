# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for native model exporter used in getitune."""

from __future__ import annotations

import logging as log
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import onnx
import openvino
import torch

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.types.export import TaskLevelExportParameters
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from getitune.backend.lightning.models.base import DataInputParams, LightningModel


class LightningModelExporter(ModelExporter):
    """Exporter that uses native torch and OpenVINO conversion tools."""

    def __init__(
        self,
        task_level_export_parameters: TaskLevelExportParameters,
        data_input_params: DataInputParams,
        resize_mode: Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"] = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        via_onnx: bool = False,
        onnx_export_configuration: dict[str, Any] | None = None,
        output_names: list[str] | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        self.via_onnx = via_onnx
        self.onnx_export_configuration = onnx_export_configuration if onnx_export_configuration is not None else {}

        # Sync output_names and input_names from onnx_export_configuration if not explicitly provided
        # This ensures they are used for both ONNX export and direct OpenVINO conversion
        if output_names is None and "output_names" in self.onnx_export_configuration:
            output_names = self.onnx_export_configuration["output_names"]
        if input_names is None and "input_names" in self.onnx_export_configuration:
            input_names = self.onnx_export_configuration["input_names"]

        super().__init__(
            task_level_export_parameters=task_level_export_parameters,
            data_input_params=data_input_params,
            resize_mode=resize_mode,
            pad_value=pad_value,
            swap_rgb=swap_rgb,
            output_names=output_names,
            input_names=input_names,
        )

        if output_names is not None:
            self.onnx_export_configuration.update({"output_names": output_names})

    def to_openvino(
        self,
        model: LightningModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        In this implementation the export is done only via standard OV/ONNX tools.
        """
        input_size = self.data_input_params.as_ncwh()
        dummy_tensor = torch.rand(input_size).to(next(model.parameters()).device)

        # Keep spatial dims static, but let the batch dimension be dynamic so any batch size works at inference.
        dynamic_shape = openvino.PartialShape([-1, *input_size[1:]])

        if self.via_onnx:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_dir = Path(tmpdirname)

                self.to_onnx(
                    model,
                    tmp_dir,
                    base_model_name,
                    Precision.FP32,
                    False,
                )
                exported_model = openvino.convert_model(tmp_dir / (base_model_name + ".onnx"), input=(dynamic_shape,))
        else:
            exported_model = openvino.convert_model(model, example_input=dummy_tensor, input=(dynamic_shape,))
        exported_model = self._postprocess_openvino_model(exported_model)

        # Validate the converted model before persisting. OpenVINO's PyTorch
        # frontend will log a deserialization error on failure and then
        # silently fall back -- leaving us with a zero-op graph that passes
        # through the "Converting to OpenVINO is done." line.  Raise here
        # instead so the caller sees the real error.
        if len(exported_model.inputs) == 0 or len(exported_model.outputs) == 0:
            msg = (
                "OpenVINO conversion produced an empty model "
                f"(inputs={len(exported_model.inputs)}, outputs={len(exported_model.outputs)}). "
                "Check preceding logs for the underlying Torch/ONNX export failure."
            )
            raise RuntimeError(msg)

        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == Precision.FP16))
        log.info(
            "Converting to OpenVINO is done. (%d inputs, %d outputs) -> %s",
            len(exported_model.inputs),
            len(exported_model.outputs),
            save_path,
        )

        return Path(save_path)

    def to_onnx(
        self,
        model: LightningModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Export the given PyTorch model to ONNX format and save it to the specified output directory.

        Args:
            model (LightningModel): The PyTorch model to be exported.
            output_dir (Path): The directory where the ONNX model will be saved.
            base_model_name (str, optional): The base name for the exported model. Defaults to "exported_model".
            precision (Precision, optional): The precision type for the exported model.
                Defaults to Precision.FP32.
            embed_metadata (bool, optional): Whether to embed metadata in the ONNX model. Defaults to True.

        Returns:
            Path: The path to the saved ONNX model.
        """
        dummy_tensor = torch.rand(self.data_input_params.as_ncwh()).to(next(model.parameters()).device)
        save_path = str(output_dir / (base_model_name + ".onnx"))
        torch.onnx.export(model, (dummy_tensor,), save_path, **self.onnx_export_configuration)

        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        onnx.save(onnx_model, save_path)
        log.info("Converting to ONNX is done.")

        return Path(save_path)
