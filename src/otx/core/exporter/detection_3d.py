# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for object detection 3D model exporter used in OTX."""

from __future__ import annotations

import logging as log
from pathlib import Path
from typing import TYPE_CHECKING

import onnx
import openvino
import torch

from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from otx.core.model.base import OTXModel


class OTXObjectDetection3DExporter(OTXNativeModelExporter):
    """Class definition for object detection 3D model exporter used in OTX."""

    def to_openvino(
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        In this implementation the export is done only via standard OV/ONNX tools.
        """
        device = next(model.parameters()).device
        dummy_tensor = torch.rand(self.input_size).to(device)
        dummy_calib_matrix = torch.rand(1, 3, 4).to(device)
        dummy_image_sizes = torch.tensor([self.input_size[::-1][:2]]).to(device)

        exported_model = openvino.convert_model(
            model,
            example_input={"images": dummy_tensor, "calib_matrix": dummy_calib_matrix, "img_sizes": dummy_image_sizes},
            input=(
                openvino.runtime.PartialShape(self.input_size),
                openvino.runtime.PartialShape([1, 3, 4]),
                openvino.runtime.PartialShape([1, 2]),
            ),
        )
        exported_model = self._postprocess_openvino_model(exported_model)

        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXPrecisionType.FP16))
        log.info("Converting to OpenVINO is done.")

        return Path(save_path)

    def to_onnx(
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Export the given PyTorch model to ONNX format and save it to the specified output directory.

        Args:
            model (OTXModel): The PyTorch model to be exported.
            output_dir (Path): The directory where the ONNX model will be saved.
            base_model_name (str, optional): The base name for the exported model. Defaults to "exported_model".
            precision (OTXPrecisionType, optional): The precision type for the exported model.
            Defaults to OTXPrecisionType.FP32.
            embed_metadata (bool, optional): Whether to embed metadata in the ONNX model. Defaults to True.

        Returns:
            Path: The path to the saved ONNX model.
        """
        device = next(model.parameters()).device
        dummy_tensor = torch.rand(self.input_size).to(device)
        dummy_calib_matrix = torch.rand(1, 3, 4).to(device)
        dummy_image_size = torch.tensor([[1350, 620]]).to(device)

        save_path = str(output_dir / (base_model_name + ".onnx"))

        torch.onnx.export(
            model,
            {"image": dummy_tensor, "calib_matrix": dummy_calib_matrix, "img_sizes": dummy_image_size},
            save_path,
            **self.onnx_export_configuration,
        )

        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        onnx.save(onnx_model, save_path)
        log.info("Converting to ONNX is done.")

        return Path(save_path)
