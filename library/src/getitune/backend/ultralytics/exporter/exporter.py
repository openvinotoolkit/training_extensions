# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class-based exporter for Ultralytics YOLO models.

Follows the same architecture as
:class:`~getitune.backend.lightning.exporter.native.LightningModelExporter`
— inherits from :class:`~getitune.backend.lightning.exporter.base.ModelExporter`
and reuses its metadata-embedding, preprocessing-parameter and postprocessing
helpers.

Key differences from the Lightning path:
  * The raw export is performed by ``ultralytics.YOLO.export()`` instead of
    ``torch.onnx.export`` / ``openvino.convert_model``.
  * FP16 is handled by ``openvino.save_model(compress_to_fp16=True)`` (OpenVINO)
    or ``onnxconverter_common`` (ONNX) — **no** manual Cast-node insertion.
  * The Ultralytics export always runs in **FP32**; weight compression happens
    as a post-step, matching the Lightning contract exactly.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import onnx
import openvino
import yaml

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.types.export import TaskLevelExportParameters
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from getitune.backend.lightning.models.base import DataInputParams

logger = logging.getLogger(__name__)


class _YOLOExportable(Protocol):
    """Protocol for the subset of ``ultralytics.YOLO`` used during export."""

    def export(self, **kwargs: Any) -> str | Path:  # noqa: ANN401
        """Export the model and return the produced artifact path."""

    @property
    def model(self) -> Any:  # noqa: ANN401
        """The underlying model with stride and end2end attributes."""
        ...


class UltralyticsModelExporter(ModelExporter):
    """Exporter for Ultralytics YOLO models.

    Delegates the actual model conversion to ``ultralytics.YOLO.export()``,
    then applies the same metadata-embedding and FP16-compression pipeline
    as :class:`LightningModelExporter`.

    Args:
        task_level_export_parameters: Collection of export parameters
            (model type, task type, labels, thresholds, etc.).
        data_input_params: Data input parameters for preprocessing metadata
            (mean, std, input_size).
        resize_mode: Resize strategy embedded in export metadata.
        pad_value: Padding value for letterbox resize.
        swap_rgb: Whether to embed ``reverse_input_channels=True`` in
            metadata so model_api swaps BGR→RGB at inference.  Set to
            ``False`` (default) because the Geti backend already sends RGB.
        output_names: Optional output tensor names to embed.
        input_names: Optional input tensor names to embed.
        export_nms: Whether to embed NMS in the exported graph.
    """

    def __init__(
        self,
        task_level_export_parameters: TaskLevelExportParameters,
        data_input_params: DataInputParams,
        resize_mode: Literal[
            "crop", "standard", "fit_to_window", "fit_to_window_letterbox"
        ] = "fit_to_window_letterbox",
        pad_value: int = 114,
        swap_rgb: bool = False,
        output_names: list[str] | None = None,
        input_names: list[str] | None = None,
        export_nms: bool = False,
    ) -> None:
        super().__init__(
            task_level_export_parameters=task_level_export_parameters,
            data_input_params=data_input_params,
            resize_mode=resize_mode,
            pad_value=pad_value,
            swap_rgb=swap_rgb,
            output_names=output_names,
            input_names=input_names,
        )
        self.export_nms = export_nms

    def to_openvino(  # pyrefly: ignore[bad-override]
        self,
        model: _YOLOExportable,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export a YOLO model to OpenVINO IR format.

        1. Export FP32 via ``model.export(format="openvino", half=False, end2end=False, nms=self.export_nms)``.
        2. Load the resulting OV model and apply inherited metadata embedding
           (preprocessing params from ``DataInputParams`` + ``TaskLevelExportParameters``).
        3. Save with ``compress_to_fp16=True`` when *precision* is FP16 —
           **no manual Cast nodes**.
        4. Clean up raw Ultralytics export artefacts.

        Args:
            model: Ultralytics ``YOLO`` instance to export.
            output_dir: Directory for the final ``<base_model_name>.xml`` file.
            base_model_name: Stem of the output file.
            precision: ``FP32`` or ``FP16``.

        Returns:
            Path to the exported ``.xml`` file.
        """
        imgsz = self.data_input_params.input_size[0]

        raw_result = model.export(
            format="openvino",
            imgsz=imgsz,
            half=False,
            end2end=False,
            nms=self.export_nms,
            project=str(output_dir),
            name="raw_export",
            exist_ok=True,
        )
        raw_path = Path(raw_result)
        raw_xml = self._find_xml_in_export(Path(raw_path))

        ov_model = openvino.Core().read_model(str(raw_xml))
        ov_model = self._postprocess_openvino_model(ov_model)

        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{base_model_name}.xml"
        openvino.save_model(ov_model, str(save_path), compress_to_fp16=(precision == Precision.FP16))

        logger.info(
            f"Ultralytics OpenVINO export done ({len(ov_model.inputs)} inputs, "
            f"{len(ov_model.outputs)} outputs) -> {save_path}"
        )

        self._cleanup_raw_export(raw_path, save_path.parent)
        self._write_metadata_yaml(model, output_dir, precision)

        return save_path

    def to_onnx(  # pyrefly: ignore[bad-override]
        self,
        model: _YOLOExportable,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Export a YOLO model to ONNX format.

        1. Export FP32 via ``model.export(format="onnx", half=False, end2end=False, nms=self.export_nms)``.
        2. Load ONNX, apply inherited metadata embedding + FP16 conversion
           (via ``onnxconverter_common``, same as Lightning).
        3. Save to target location.

        Args:
            model: Ultralytics ``YOLO`` instance to export.
            output_dir: Directory for the final ``.onnx`` file.
            base_model_name: Stem of the output file.
            precision: ``FP32`` or ``FP16``.
            embed_metadata: Whether to embed metadata into ONNX model.

        Returns:
            Path to the exported ``.onnx`` file.
        """
        imgsz = self.data_input_params.input_size[0]

        raw_result = model.export(
            format="onnx",
            imgsz=imgsz,
            half=False,
            end2end=False,
            nms=self.export_nms,
            project=str(output_dir),
            name="raw_export",
            exist_ok=True,
        )
        raw_path = Path(raw_result)

        onnx_model = onnx.load(str(raw_path))
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f"{base_model_name}.onnx"
        onnx.save(onnx_model, str(save_path))

        logger.info(f"Ultralytics ONNX export done -> {save_path}")

        if raw_path.resolve() != save_path.resolve():
            raw_path.unlink(missing_ok=True)

        self._write_metadata_yaml(model, output_dir, precision)

        return save_path

    def _write_metadata_yaml(self, model: _YOLOExportable, output_dir: Path, precision: Precision) -> None:
        """Write Ultralytics-compatible ``metadata.yaml`` alongside the exported model.

        This file enables Ultralytics CLI usage with the exported model::

            yolo predict model=./output_dir/ source=image.jpg
            yolo val model=./output_dir/ data=dataset.yaml

        The format matches what ``ultralytics.engine.exporter.Exporter`` produces,
        so the CLI treats our export identically to a native Ultralytics export.

        Args:
            model: The Ultralytics YOLO model (provides stride).
            output_dir: Directory where the exported model files live.
            precision: Export precision (affects ``half`` field in args).
        """
        import ultralytics

        params = self.task_level_export_parameters
        label_names = params.label_info.label_names if params.label_info else []
        names = dict(enumerate(label_names))

        # Map internal task_type to Ultralytics task identifier.
        task_map = {
            "detection": "detect",
            "instance_segmentation": "segment",
            "classification": "classify",
            "segmentation": "semantic",
        }
        task = task_map.get(params.task_type, params.task_type)

        # Get stride from the model dynamically.
        stride = 32
        try:
            model_stride = model.model.stride
            stride = int(max(model_stride)) if hasattr(model_stride, "__iter__") else int(model_stride)
        except (AttributeError, TypeError, ValueError):
            pass

        imgsz = list(self.data_input_params.input_size)

        metadata: dict[str, Any] = {
            "description": f"Ultralytics {params.model_name} model",
            "author": "Ultralytics",
            "date": datetime.now(tz=timezone.utc).isoformat(),
            "version": ultralytics.__version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": stride,
            "task": task,
            "batch": 1,
            "imgsz": imgsz,
            "names": names,
            "channels": 3,
            "end2end": False,
            "args": {
                "data": None,
                "batch": 1,
                "fraction": 1.0,
                "half": precision == Precision.FP16,
                "int8": False,
                "dynamic": False,
                "nms": self.export_nms,
            },
        }

        yaml_path = output_dir / "metadata.yaml"
        with open(yaml_path, "w") as f:  # noqa: PTH123
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Wrote Ultralytics metadata: {yaml_path}")

    @staticmethod
    def _find_xml_in_export(export_path: Path) -> Path:
        """Locate the ``.xml`` file inside an Ultralytics OpenVINO export.

        Ultralytics typically writes to a directory (e.g. ``model_openvino/``).
        This helper finds the first ``.xml`` file inside that directory, or
        returns *export_path* itself if it is already an XML file.
        """
        if export_path.is_file() and export_path.suffix == ".xml":
            return export_path

        if export_path.is_dir():
            xml_files = list(export_path.glob("*.xml"))
            if xml_files:
                return xml_files[0]

        msg = f"No .xml file found in Ultralytics export output: {export_path}"
        raise FileNotFoundError(msg)

    @staticmethod
    def _cleanup_raw_export(raw_path: Path, target_dir: Path) -> None:
        """Remove Ultralytics raw export artefacts if they differ from target.

        Ultralytics creates a directory like ``runs/detect/export/model_openvino/``.
        After we have copied the final model to *target_dir*, remove the raw
        directory to avoid stale artefacts.
        """
        if not raw_path.exists():
            return

        raw_resolved = raw_path.resolve()
        target_resolved = target_dir.resolve()

        # Don't delete if the raw export IS the target
        if raw_resolved == target_resolved:
            return
        # Don't delete if the raw export is a parent of the target
        if target_resolved.is_relative_to(raw_resolved):
            return

        if raw_path.is_dir():
            shutil.rmtree(raw_path, ignore_errors=True)
        elif raw_path.is_file():
            raw_path.unlink(missing_ok=True)
