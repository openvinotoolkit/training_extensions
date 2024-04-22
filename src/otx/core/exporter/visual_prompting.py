# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model exporter used in OTX."""

from __future__ import annotations

import logging as log
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import onnx
import openvino
import torch

from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from otx.core.model.base import OTXModel


class OTXVisualPromptingModelExporter(OTXNativeModelExporter):
    """Exporter for visual prompting models that uses native torch and OpenVINO conversion tools."""

    def export(  # type: ignore[override]
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        export_format: OTXExportFormatType = OTXExportFormatType.OPENVINO,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> dict[str, Path]:
        """Exports input model to the specified deployable format, such as OpenVINO IR or ONNX.

        Args:
            model (OTXModel): OTXModel to be exported
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            format (OTXExportFormatType): final format of the exported model
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

        Returns:
            dict[str, Path]: paths to the exported models
        """
        # NOTE: Rather than using OTXModel.forward_for_tracing()
        # Use the nested `image_encoder` and `decoder` models' forward functions directly
        models: dict[str, torch.nn.Module] = {
            "image_encoder": model.model.image_encoder,
            "decoder": model.model,
        }

        if export_format == OTXExportFormatType.OPENVINO:
            fn = self.to_openvino
        elif export_format == OTXExportFormatType.ONNX:
            fn = self.to_onnx
        elif export_format == OTXExportFormatType.EXPORTABLE_CODE:
            msg = "exportable code will be supported soon."
            raise NotImplementedError(msg)
        else:
            msg = f"Unsupported export format: {export_format}"
            raise ValueError(msg)

        return {  # type: ignore[return-value]
            module: fn(models[module], output_dir, f"{base_model_name}_{module}", precision)
            for module in ["image_encoder", "decoder"]
        }

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
        if not self.via_onnx:
            log.info("openvino export on OTXVisualPromptingModelExporter supports only via_onnx, set to True.")

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_dir = Path(tmpdirname)

            self.to_onnx(
                model,
                tmp_dir,
                base_model_name,
                OTXPrecisionType.FP32,
                False,
            )
            exported_model = openvino.convert_model(tmp_dir / (base_model_name + ".onnx"))

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
            **self.get_onnx_dummy_inputs(base_model_name, model),  # type: ignore[arg-type]
            **self.onnx_export_configuration,
        )

        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        onnx.save(onnx_model, save_path)
        log.info("Converting to ONNX is done.")

        return Path(save_path)

    def get_onnx_dummy_inputs(
        self,
        base_model_name: Literal["exported_model_image_encoder", "exported_model_decoder"],
        model: torch.nn.Module,
    ) -> dict[str, Any]:
        """Get onnx dummy inputs."""
        if base_model_name == "exported_model_image_encoder":
            dummy_inputs = {"images": torch.randn(self.input_size, dtype=torch.float32)}
            output_names = ["image_embeddings"]
            dynamic_axes = None
        else:
            dummy_inputs = {
                "image_embeddings": torch.zeros(
                    1,
                    model.embed_dim,
                    model.image_embedding_size,
                    model.image_embedding_size,
                    dtype=torch.float32,
                ),
                "point_coords": torch.randint(low=0, high=1024, size=(1, 2, 2), dtype=torch.float32),
                "point_labels": torch.randint(low=0, high=4, size=(1, 2), dtype=torch.float32),
                "mask_input": torch.randn(
                    1,
                    1,
                    4 * model.image_embedding_size,
                    4 * model.image_embedding_size,
                    dtype=torch.float32,
                ),
                "has_mask_input": torch.tensor([[1]], dtype=torch.float32),
                "orig_size": torch.randint(low=256, high=2048, size=(1, 2), dtype=torch.int64),
            }
            output_names = ["upscaled_masks", "iou_predictions", "low_res_masks"]
            dynamic_axes = {"point_coords": {1: "num_points"}, "point_labels": {1: "num_points"}}

        return {
            "args": tuple(dummy_inputs.values()),
            "input_names": list(dummy_inputs.keys()),
            "output_names": output_names,
            "dynamic_axes": dynamic_axes,
        }
