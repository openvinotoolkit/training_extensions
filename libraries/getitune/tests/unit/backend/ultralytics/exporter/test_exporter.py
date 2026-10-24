# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics model exporter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.exporter import UltralyticsModelExporter
from getitune.config.data import IntensityConfig
from getitune.types.export import ExportFormat, TaskLevelExportParameters
from getitune.types.label import LabelInfo
from getitune.types.precision import Precision


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=["cat", "dog"], label_ids=["0", "1"], label_groups=[["cat", "dog"]])


def _data_input_params(imgsz: int = 640) -> DataInputParams:
    return DataInputParams(
        input_size=(imgsz, imgsz),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        intensity_config=IntensityConfig(mode="scale_to_unit", storage_dtype="uint8"),
    )


def _export_parameters() -> TaskLevelExportParameters:
    return TaskLevelExportParameters(
        model_type="YOLO11",
        model_name="yolo26_n",
        task_type="detection",
        label_info=_label_info(),
        optimization_config={},
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )


def _make_exporter(**kwargs) -> UltralyticsModelExporter:
    """Create an exporter with sensible defaults."""
    defaults = {
        "task_level_export_parameters": _export_parameters(),
        "data_input_params": _data_input_params(),
    }
    defaults.update(kwargs)
    return UltralyticsModelExporter(**defaults)


class TestUltralyticsModelExporterInit:
    """Tests for exporter construction and default values."""

    def test_default_resize_mode(self) -> None:
        exporter = _make_exporter()
        assert exporter.resize_mode == "fit_to_window_letterbox"

    def test_default_pad_value(self) -> None:
        exporter = _make_exporter()
        assert exporter.pad_value == 114

    def test_default_swap_rgb(self) -> None:
        exporter = _make_exporter()
        assert exporter.swap_rgb is False

    def test_inherits_from_model_exporter(self) -> None:
        from getitune.backend.lightning.exporter.base import ModelExporter

        exporter = _make_exporter()
        assert isinstance(exporter, ModelExporter)


class TestExporterMetadata:
    """Tests for metadata produced by the inherited metadata pipeline."""

    def test_metadata_includes_model_type(self) -> None:
        exporter = _make_exporter()
        metadata = exporter.metadata
        assert ("model_info", "model_type") in metadata
        assert metadata[("model_info", "model_type")] == "YOLO11"

    def test_metadata_includes_task_type(self) -> None:
        exporter = _make_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "task_type")] == "detection"

    def test_metadata_includes_labels(self) -> None:
        exporter = _make_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "labels")] == "cat dog"
        assert metadata[("model_info", "label_ids")] == "0 1"

    def test_metadata_includes_thresholds(self) -> None:
        exporter = _make_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "confidence_threshold")] == "0.25"
        assert metadata[("model_info", "iou_threshold")] == "0.5"

    def test_metadata_includes_getitune_version(self) -> None:
        import getitune

        exporter = _make_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "getitune_version")] == getitune.__version__

    def test_extended_metadata_includes_preprocessing(self) -> None:
        """_extend_model_metadata should add mean/scale/resize from DataInputParams."""
        exporter = _make_exporter()
        extended = exporter._extend_model_metadata(exporter.metadata)

        assert extended[("model_info", "mean_values")] == "0.0 0.0 0.0"
        assert extended[("model_info", "scale_values")] == "1.0 1.0 1.0"
        assert extended[("model_info", "resize_type")] == "fit_to_window_letterbox"
        assert extended[("model_info", "pad_value")] == "114"
        assert extended[("model_info", "reverse_input_channels")] == "False"
        assert extended[("model_info", "input_dtype")] == "u8"
        assert extended[("model_info", "intensity_mode")] == "scale_to_unit"
        assert ("model_info", "intensity_max_value") not in extended

    def test_postprocess_openvino_embeds_intensity_metadata(self) -> None:
        """OpenVINO postprocessing should write the IntensityConfig contract into rt_info."""
        exporter = _make_exporter()
        mock_ov_model = MagicMock()
        mock_ov_model.outputs = []
        mock_ov_model.inputs = []

        exporter._postprocess_openvino_model(mock_ov_model)

        rt_info = {tuple(call.args[1]): call.args[0] for call in mock_ov_model.set_rt_info.call_args_list}
        assert rt_info[("model_info", "scale_values")] == "1.0 1.0 1.0"
        assert rt_info[("model_info", "input_dtype")] == "u8"
        assert rt_info[("model_info", "intensity_mode")] == "scale_to_unit"

    def test_all_metadata_values_are_strings(self) -> None:
        exporter = _make_exporter()
        extended = exporter._extend_model_metadata(exporter.metadata)
        for key, value in extended.items():
            assert isinstance(value, str), f"Value for {key} is {type(value)}, expected str"

    def test_custom_thresholds_via_wrap(self) -> None:
        """Threshold overrides via TaskLevelExportParameters.wrap should propagate."""
        params = _export_parameters().wrap(confidence_threshold=0.5, iou_threshold=0.6)
        exporter = _make_exporter(task_level_export_parameters=params)
        metadata = exporter.metadata
        assert metadata[("model_info", "confidence_threshold")] == "0.5"
        assert metadata[("model_info", "iou_threshold")] == "0.6"


class TestToOpenvino:
    """Tests for the to_openvino export path."""

    @pytest.mark.parametrize("export_nms", [False, True])
    def test_exports_fp32_with_correct_args(self, tmp_path: Path, export_nms: bool) -> None:
        """Should forward graph NMS without overriding end-to-end mode."""
        exporter = _make_exporter(export_nms=export_nms)

        # Setup: Ultralytics produces a directory with a .xml
        raw_dir = tmp_path / "raw_openvino"
        raw_dir.mkdir()
        (raw_dir / "model.xml").touch()
        (raw_dir / "model.bin").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model") as mock_save,
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model),
        ):
            output_dir = tmp_path / "output"
            result = exporter.to_openvino(mock_yolo, output_dir, "exported_model", Precision.FP32)

        mock_yolo.export.assert_called_once_with(
            format="openvino",
            imgsz=640,
            half=False,
            end2end=False,
            nms=export_nms,
            project=str(tmp_path / "output"),
            name="raw_export",
            exist_ok=True,
        )
        mock_save.assert_called_once_with(mock_ov_model, str(output_dir / "exported_model.xml"), compress_to_fp16=False)
        assert result == output_dir / "exported_model.xml"

    def test_fp16_uses_compress_to_fp16(self, tmp_path: Path) -> None:
        """FP16 should use compress_to_fp16=True, NOT half=True."""
        exporter = _make_exporter()

        raw_dir = tmp_path / "raw_openvino"
        raw_dir.mkdir()
        (raw_dir / "model.xml").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model") as mock_save,
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model),
        ):
            output_dir = tmp_path / "output"
            result = exporter.to_openvino(mock_yolo, output_dir, "exported_model", Precision.FP16)

        # Ultralytics should still get half=False (always FP32 export)
        _, call_kwargs = mock_yolo.export.call_args
        assert call_kwargs["half"] is False

        # OpenVINO save should use compress_to_fp16=True
        mock_save.assert_called_once_with(mock_ov_model, str(output_dir / "exported_model.xml"), compress_to_fp16=True)
        assert result == output_dir / "exported_model.xml"

    def test_postprocess_is_called(self, tmp_path: Path) -> None:
        """_postprocess_openvino_model should be called to embed metadata."""
        exporter = _make_exporter()

        raw_dir = tmp_path / "raw_openvino"
        raw_dir.mkdir()
        (raw_dir / "model.xml").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model"),
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model) as mock_pp,
        ):
            exporter.to_openvino(mock_yolo, tmp_path / "out", "model", Precision.FP32)

        mock_pp.assert_called_once_with(mock_ov_model)

    def test_no_xml_raises_file_not_found(self, tmp_path: Path) -> None:
        """If Ultralytics export produces a dir with no .xml, should raise."""
        exporter = _make_exporter()

        empty_dir = tmp_path / "empty_export"
        empty_dir.mkdir()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(empty_dir)

        with pytest.raises(FileNotFoundError, match="No .xml file found"):
            exporter.to_openvino(mock_yolo, tmp_path / "out", "model", Precision.FP32)

    def test_cleanup_removes_raw_export(self, tmp_path: Path) -> None:
        """Raw Ultralytics export directory should be cleaned up."""
        exporter = _make_exporter()

        raw_dir = tmp_path / "raw_openvino"
        raw_dir.mkdir()
        (raw_dir / "model.xml").touch()
        (raw_dir / "model.bin").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        output_dir = tmp_path / "output"
        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model"),
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model),
        ):
            exporter.to_openvino(mock_yolo, output_dir, "exported_model", Precision.FP32)

        # Raw dir should have been cleaned up
        assert not raw_dir.exists()


class TestToOnnx:
    """Tests for the to_onnx export path."""

    @pytest.mark.parametrize("export_nms", [False, True])
    def test_exports_fp32_with_correct_args(self, tmp_path: Path, export_nms: bool) -> None:
        exporter = _make_exporter(export_nms=export_nms)

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)

        mock_onnx_model = MagicMock()

        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save") as mock_save,
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model) as mock_pp,
        ):
            output_dir = tmp_path / "output"
            result = exporter.to_onnx(mock_yolo, output_dir, "exported_model", Precision.FP32)

        mock_yolo.export.assert_called_once_with(
            format="onnx",
            imgsz=640,
            half=False,
            end2end=False,
            nms=export_nms,
            project=str(tmp_path / "output"),
            name="raw_export",
            exist_ok=True,
        )
        mock_pp.assert_called_once_with(mock_onnx_model, True, Precision.FP32)
        mock_save.assert_called_once_with(mock_onnx_model, str(output_dir / "exported_model.onnx"))
        assert result == output_dir / "exported_model.onnx"

    def test_fp16_passes_precision_to_postprocess(self, tmp_path: Path) -> None:
        exporter = _make_exporter()

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)

        mock_onnx_model = MagicMock()

        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save"),
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model) as mock_pp,
        ):
            exporter.to_onnx(mock_yolo, tmp_path / "out", "model", Precision.FP16)

        # Ultralytics still exports FP32
        _, call_kwargs = mock_yolo.export.call_args
        assert call_kwargs["half"] is False

        # FP16 conversion is done by _postprocess_onnx_model (inherited)
        mock_pp.assert_called_once_with(mock_onnx_model, True, Precision.FP16)

    def test_raw_onnx_cleaned_up(self, tmp_path: Path) -> None:
        """Raw ONNX file should be removed when it differs from the output."""
        exporter = _make_exporter()

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.write_bytes(b"fake")

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)

        mock_onnx_model = MagicMock()

        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save"),
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model),
        ):
            output_dir = tmp_path / "output"
            exporter.to_onnx(mock_yolo, output_dir, "model", Precision.FP32)

        assert not raw_onnx.exists()


class TestExportDispatch:
    """Tests for the inherited export() dispatch method."""

    def test_openvino_dispatches_to_to_openvino(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        mock_yolo = MagicMock()

        with patch.object(exporter, "to_openvino", return_value=tmp_path / "model.xml") as mock_to_ov:
            result = exporter.export(mock_yolo, tmp_path, "model", ExportFormat.OPENVINO, Precision.FP32)

        mock_to_ov.assert_called_once_with(mock_yolo, tmp_path, "model", Precision.FP32)
        assert result == tmp_path / "model.xml"

    def test_onnx_dispatches_to_to_onnx(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        mock_yolo = MagicMock()

        with patch.object(exporter, "to_onnx", return_value=tmp_path / "model.onnx") as mock_to_onnx:
            result = exporter.export(mock_yolo, tmp_path, "model", ExportFormat.ONNX, Precision.FP32)

        mock_to_onnx.assert_called_once_with(mock_yolo, tmp_path, "model", Precision.FP32)
        assert result == tmp_path / "model.onnx"

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        mock_yolo = MagicMock()

        bad_format = MagicMock(spec=ExportFormat)
        bad_format.value = "TORCHSCRIPT"

        with pytest.raises(ValueError, match="Unsupported export format"):
            exporter.export(mock_yolo, tmp_path, "model", bad_format, Precision.FP32)


class TestFindXmlInExport:
    """Tests for _find_xml_in_export static method."""

    def test_returns_xml_file_directly(self, tmp_path: Path) -> None:
        xml = tmp_path / "model.xml"
        xml.touch()
        assert UltralyticsModelExporter._find_xml_in_export(xml) == xml

    def test_finds_xml_in_directory(self, tmp_path: Path) -> None:
        (tmp_path / "model.xml").touch()
        result = UltralyticsModelExporter._find_xml_in_export(tmp_path)
        assert result.suffix == ".xml"

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No .xml file found"):
            UltralyticsModelExporter._find_xml_in_export(empty)


class TestCleanupRawExport:
    """Tests for _cleanup_raw_export static method."""

    def test_removes_different_directory(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "file.txt").touch()
        target = tmp_path / "target"
        target.mkdir()

        UltralyticsModelExporter._cleanup_raw_export(raw, target)
        assert not raw.exists()

    def test_does_not_remove_same_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "output"
        target.mkdir()
        (target / "file.txt").touch()

        UltralyticsModelExporter._cleanup_raw_export(target, target)
        assert target.exists()

    def test_does_not_remove_parent_of_target(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        target = raw / "sub"
        target.mkdir()

        UltralyticsModelExporter._cleanup_raw_export(raw, target)
        assert raw.exists()

    def test_handles_nonexistent_path(self, tmp_path: Path) -> None:
        """Should not raise for nonexistent raw path."""
        UltralyticsModelExporter._cleanup_raw_export(tmp_path / "nonexistent", tmp_path)

    def test_model_type_in_is_export_parameters(self) -> None:
        """IS model should report model_type='YOLO-seg'."""
        from getitune.backend.ultralytics.models.instance_segmentation import UltralyticsInstSegModel
        from getitune.types.label import LabelInfo

        label_info = LabelInfo(label_names=["a"], label_ids=["0"], label_groups=[["a"]])
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", label_info=label_info)
        params = model._export_parameters
        assert params.model_type == "YOLO-seg"
        assert params.task_type == "instance_segmentation"


class TestMetadataYaml:
    """Tests for Ultralytics metadata.yaml generation."""

    def test_openvino_export_generates_metadata_yaml(self, tmp_path: Path) -> None:
        """to_openvino() should produce metadata.yaml alongside the model."""
        exporter = _make_exporter()

        raw_dir = tmp_path / "output" / "raw_export"
        raw_dir.mkdir(parents=True)
        raw_xml = raw_dir / "model.xml"
        raw_xml.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)
        mock_yolo.model.stride = [8, 16, 32]

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]
        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        output_dir = tmp_path / "output"
        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model"),
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model),
        ):
            exporter.to_openvino(mock_yolo, output_dir, "exported_model", Precision.FP32)

        metadata_path = output_dir / "metadata.yaml"
        assert metadata_path.exists()

        import yaml

        with metadata_path.open() as f:
            metadata = yaml.safe_load(f)

        assert metadata["task"] == "detect"
        assert metadata["stride"] == 32
        assert metadata["imgsz"] == [640, 640]
        assert metadata["names"] == {0: "cat", 1: "dog"}
        assert metadata["end2end"] is False
        assert metadata["channels"] == 3
        assert metadata["batch"] == 1
        assert metadata["author"] == "Ultralytics"
        assert metadata["args"]["half"] is False
        assert metadata["args"]["int8"] is False

    def test_onnx_export_generates_metadata_yaml(self, tmp_path: Path) -> None:
        """to_onnx() should produce metadata.yaml alongside the model."""
        exporter = _make_exporter()

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)
        mock_yolo.model.stride = [8, 16, 32]

        mock_onnx_model = MagicMock()

        output_dir = tmp_path / "output"
        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save"),
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model),
        ):
            exporter.to_onnx(mock_yolo, output_dir, "model", Precision.FP32)

        metadata_path = output_dir / "metadata.yaml"
        assert metadata_path.exists()

        import yaml

        with metadata_path.open() as f:
            metadata = yaml.safe_load(f)

        assert metadata["task"] == "detect"
        assert metadata["names"] == {0: "cat", 1: "dog"}

    def test_fp16_export_sets_half_true(self, tmp_path: Path) -> None:
        """FP16 export should set args.half=true in metadata."""
        exporter = _make_exporter()

        raw_dir = tmp_path / "output" / "raw_export"
        raw_dir.mkdir(parents=True)
        raw_xml = raw_dir / "model.xml"
        raw_xml.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)
        mock_yolo.model.stride = [8, 16, 32]

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]
        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        output_dir = tmp_path / "output"
        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model"),
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model),
        ):
            exporter.to_openvino(mock_yolo, output_dir, "model", Precision.FP16)

        import yaml

        with (output_dir / "metadata.yaml").open() as f:
            metadata = yaml.safe_load(f)

        assert metadata["args"]["half"] is True

    def test_instance_segmentation_task_mapping(self, tmp_path: Path) -> None:
        """Instance segmentation task_type should map to 'segment' in metadata."""
        seg_params = TaskLevelExportParameters(
            model_type="YOLO-seg",
            model_name="yolo26n-seg",
            task_type="instance_segmentation",
            label_info=_label_info(),
            optimization_config={},
            confidence_threshold=0.25,
            iou_threshold=0.5,
        )
        exporter = _make_exporter(task_level_export_parameters=seg_params)

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)
        mock_yolo.model.stride = [8, 16, 32]

        mock_onnx_model = MagicMock()

        output_dir = tmp_path / "output"
        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save"),
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model),
        ):
            exporter.to_onnx(mock_yolo, output_dir, "model", Precision.FP32)

        import yaml

        with (output_dir / "metadata.yaml").open() as f:
            metadata = yaml.safe_load(f)

        assert metadata["task"] == "segment"

    def test_semantic_segmentation_task_mapping(self, tmp_path: Path) -> None:
        """Semantic segmentation task_type should map to 'semantic' in metadata."""
        seg_params = TaskLevelExportParameters(
            model_type="YOLO-sem",
            model_name="yolo26n-sem",
            task_type="segmentation",
            label_info=_label_info(),
            optimization_config={},
            confidence_threshold=0.0,
            return_soft_prediction=True,
            blur_strength=0,
        )
        exporter = _make_exporter(task_level_export_parameters=seg_params)

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)
        mock_yolo.model.stride = [8]

        mock_onnx_model = MagicMock()

        output_dir = tmp_path / "output"
        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save"),
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model),
        ):
            exporter.to_onnx(mock_yolo, output_dir, "model", Precision.FP32)

        import yaml

        with (output_dir / "metadata.yaml").open() as f:
            metadata = yaml.safe_load(f)

        assert metadata["task"] == "semantic"

    def test_stride_from_model(self, tmp_path: Path) -> None:
        """Stride should be read dynamically from model.model.stride."""
        exporter = _make_exporter()

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)
        mock_yolo.model.stride = [4, 8, 16, 32, 64]

        mock_onnx_model = MagicMock()

        output_dir = tmp_path / "output"
        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save"),
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model),
        ):
            exporter.to_onnx(mock_yolo, output_dir, "model", Precision.FP32)

        import yaml

        with (output_dir / "metadata.yaml").open() as f:
            metadata = yaml.safe_load(f)

        assert metadata["stride"] == 64
