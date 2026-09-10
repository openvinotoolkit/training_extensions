# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Native torch/OpenVINO exporter for the Hugging Face backend."""

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
    from getitune.backend.huggingface.models.base import HFModel
    from getitune.backend.lightning.models.base import DataInputParams

__all__ = ["HFModelExporter"]


class HFModelExporter(ModelExporter):
    """Export HuggingFace models from PyTorch to ONNX and OpenVINO."""

    def __init__(
        self,
        task_level_export_parameters: TaskLevelExportParameters,
        data_input_params: DataInputParams,
        resize_mode: Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"] = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        onnx_export_configuration: dict[str, Any] | None = None,
        output_names: list[str] | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        self.onnx_export_configuration = dict(onnx_export_configuration or {})
        self.onnx_export_configuration.setdefault("dynamo", True)
        self.onnx_export_configuration.setdefault("do_constant_folding", True)
        self.onnx_export_configuration.setdefault("opset_version", 17)

        if output_names is None:
            output_names = self.onnx_export_configuration.get("output_names")
        if input_names is None:
            input_names = self.onnx_export_configuration.get("input_names")

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
            self.onnx_export_configuration.setdefault("output_names", output_names)
        if input_names is not None:
            self.onnx_export_configuration.setdefault("input_names", input_names)

    def to_openvino(  # pyrefly: ignore[bad-override]
        self,
        model: HFModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export to OpenVINO IR.

        Tries a direct torch -> OpenVINO conversion first, with no intermediate
        artifact. If that fails for a particular model (some architectures don't
        trace cleanly through OpenVINO's torch frontend), it falls back to the
        legacy ONNX route: ``torch.onnx.export``.
        """
        input_size = self.data_input_params.as_ncwh()
        dynamic_shape = openvino.PartialShape([-1, *input_size[1:]])
        dummy_input = torch.rand(input_size).to(next(model.parameters()).device)

        try:
            exported_model = openvino.convert_model(model, example_input=dummy_input, input=(dynamic_shape,))
            log.info("Direct torch -> OpenVINO conversion succeeded.")
        except Exception as direct_error:
            log.warning(
                "Direct torch -> OpenVINO conversion failed (%s); falling back to ONNX. "
                "If this is unexpected, report it so the model can be added to an allow-list.",
                direct_error,
            )
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                tmp_dir = Path(tmp_dir_name)
                self.to_onnx(model, tmp_dir, base_model_name, Precision.FP32, embed_metadata=False)
                exported_model = openvino.convert_model(tmp_dir / (base_model_name + ".onnx"), input=(dynamic_shape,))

        exported_model = self._postprocess_openvino_model(exported_model)

        if len(exported_model.inputs) == 0 or len(exported_model.outputs) == 0:
            msg = (
                "OpenVINO conversion produced an empty model "
                f"(inputs={len(exported_model.inputs)}, outputs={len(exported_model.outputs)}). "
                "Check preceding logs for the underlying conversion failure."
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

    def to_onnx(  # pyrefly: ignore[bad-override]
        self,
        model: HFModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Trace *model* and save it as ONNX."""
        dummy_tensor = torch.rand(self.data_input_params.as_ncwh()).to(next(model.parameters()).device)
        save_path = str(output_dir / (base_model_name + ".onnx"))
        torch.onnx.export(model, (dummy_tensor,), save_path, **self.onnx_export_configuration)

        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        onnx.save(onnx_model, save_path)
        log.info("Converting to ONNX is done.")
        return Path(save_path)
