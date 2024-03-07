# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for visual prompting model exporter used in OTX."""

from __future__ import annotations

import json
import logging as log
import os
import tempfile
from pathlib import Path
from typing import Any, Literal
from zipfile import ZipFile

import onnx
import openvino
import torch

from otx.core.exporter.exportable_code import demo
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType


class OTXVisualPromptingModelExporter(OTXNativeModelExporter):
    """Exporter for visual prompting models that uses native torch and OpenVINO conversion tools."""

    def export(  # type: ignore[override]
        self,
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        export_format: OTXExportFormatType = OTXExportFormatType.OPENVINO,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path | dict[str, Path]:
        """Exports input model to the specified deployable format, such as OpenVINO IR or ONNX.

        Args:
            model (torch.nn.Module): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            format (OTXExportFormatType): final format of the exported model
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

        Returns:
            (Path, dict[str, Path]): path(s) to the exported model(s)
        """
        models: dict[str, torch.nn.Module] = {
            "image_encoder": model.image_encoder,
            "decoder": model,
        }

        if export_format == OTXExportFormatType.OPENVINO:
            return {
                module: self.to_openvino(models[module], output_dir, f"{base_model_name}_{module}", precision)
                for module in ["image_encoder", "decoder"]
            }
        if export_format == OTXExportFormatType.ONNX:
            return {
                module: self.to_onnx(models[module], output_dir, f"{base_model_name}_{module}", precision)
                for module in ["image_encoder", "decoder"]
            }
        if export_format == OTXExportFormatType.EXPORTABLE_CODE:
            return self.to_exportable_code(models, output_dir, base_model_name, precision)

        msg = f"Unsupported export format: {export_format}"
        raise ValueError(msg)

    def to_openvino(
        self,
        model: torch.nn.Module,
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
        model: torch.nn.Module,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Export the given PyTorch model to ONNX format and save it to the specified output directory.

        Args:
            model (torch.nn.Module): The PyTorch model to be exported.
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

    def to_exportable_code(
        self,
        model: dict[str, torch.nn.Module],
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export to zip folder final OV IR model with runable demo.

        Args:
            model (dict[str, torch.nn.Module]): pytorch model top export
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

        Returns:
            Path: path to the exported model.
        """
        work_dir = Path(demo.__file__).parent
        parameters: dict[str, Any] = {}
        if self.metadata is not None:
            parameters["type_of_model"] = self.metadata.get(("model_info", "task_type"), "")
            parameters["converter_type"] = self.metadata.get(("model_info", "model_type"), "")
            parameters["model_parameters"] = {
                "labels": self.metadata.get(("model_info", "labels"), ""),
                "labels_ids": self.metadata.get(("model_info", "label_ids"), ""),
            }

        output_zip_path = output_dir / "exportable_code.zip"
        Path.mkdir(output_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir, ZipFile(output_zip_path, "x") as arch:
            # model files
            for module in ["image_encoder", "decoder"]:
                path_to_model = self.to_openvino(
                    model[module],
                    Path(temp_dir),
                    f"{base_model_name}_{module}",
                    precision,
                )
                arch.write(str(path_to_model), Path("model") / f"{module}.xml")
                arch.write(path_to_model.with_suffix(".bin"), Path("model") / f"{module}.bin")

            arch.writestr(
                str(Path("model") / "config.json"),
                json.dumps(parameters, ensure_ascii=False, indent=4),
            )
            # python files
            arch.write(
                work_dir / "requirements.txt",
                Path("python") / "requirements.txt",
            )
            arch.write(work_dir.parents[5] / "LICENSE", Path("python") / "LICENSE")
            arch.write(work_dir / "demo.py", Path("python") / "demo.py")
            arch.write(work_dir / "README.md", Path("./") / "README.md")
            arch.write(work_dir / "setup.py", Path("python") / "setup.py")
            # write demo_package
            demo_package = work_dir / "demo_package"
            for root, _, files in os.walk(demo_package):
                if root.endswith("__pycache__"):
                    continue
                for file in files:
                    file_path = Path(root) / file
                    archive_path = file_path.relative_to(demo_package)
                    arch.write(file_path, Path("python") / "demo_package" / archive_path)
        return output_zip_path

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
