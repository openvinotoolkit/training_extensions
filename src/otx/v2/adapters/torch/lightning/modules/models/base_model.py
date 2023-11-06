"""Lightning base OTX Model."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from otx.v2.adapters.torch.modules.models.base_model import BaseOTXModel

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from pytorch_lightning.trainer.connectors.accelerator_connector import (
        _PRECISION_INPUT,
    )


class BaseOTXLightningModel(BaseOTXModel):
    """OTX base model class for OTX Lightning models.

    This class defines the interface for OTX Lightning models, including the callbacks to be used during training
    and the ability to export the model to a specified format (ONNX & OPENVINO).

    Attributes:
        callbacks (list[Callback]): A list of callbacks to be used during training.
    """

    config: DictConfig
    device: torch.device

    def export(
        self,
        export_dir: str | Path,
        export_type: str = "OPENVINO",
        precision: _PRECISION_INPUT | None = None,
    ) -> dict:
        """Export the model to a specified format.

        Args:
            export_dir (str | Path): The directory to export the model to.
            export_type (str, optional): The type of export to perform. Defaults to "OPENVINO".
            precision (_PRECISION_INPUT | None, optional): The precision to use for the export. Defaults to None.

        Returns:
            dict: A dictionary containing information about the exported model.
        """
        Path(export_dir).mkdir(exist_ok=True, parents=True)

        # Torch to onnx
        onnx_dir = Path(export_dir) / "onnx"
        onnx_dir.mkdir(exist_ok=True, parents=True)
        onnx_model = str(onnx_dir / "onnx_model.onnx")

        height, width = self.config.model.get("input_size", (256, 256))
        torch.onnx.export(
            model=self,
            args=torch.zeros((1, 3, height, width)).to(self.device),
            f=onnx_model,
            opset_version=11,
        )

        results: dict = {"outputs": {}}
        results["outputs"]["onnx"] = onnx_model

        if export_type.upper() == "OPENVINO":
            # ONNX to IR
            from subprocess import run

            ir_dir = Path(export_dir) / "openvino"
            ir_dir.mkdir(exist_ok=True, parents=True)
            optimize_command = [
                "mo",
                "--input_model",
                onnx_model,
                "--output_dir",
                str(ir_dir),
                "--model_name",
                "openvino",
            ]
            if precision in ("16", 16, "fp16"):
                optimize_command.append("--compress_to_fp16")
            _ = run(args=optimize_command, check=False)
            bin_file = Path(ir_dir) / "openvino.bin"
            xml_file = Path(ir_dir) / "openvino.xml"
            if bin_file.exists() and xml_file.exists():
                results["outputs"]["bin"] = str(Path(ir_dir) / "openvino.bin")
                results["outputs"]["xml"] = str(Path(ir_dir) / "openvino.xml")
            else:
                msg = "OpenVINO Export failed."
                raise RuntimeError(msg)
        return results
