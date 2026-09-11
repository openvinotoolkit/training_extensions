# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared ``UltralyticsModel`` base behavior.

Detection/instance-segmentation model wrappers are used as concrete
stand-ins to exercise the generic behavior implemented in
``getitune.backend.ultralytics.models.base.UltralyticsModel``
(checkpoint loading, ``data_input_params``, ``_export_parameters``, and the
``_pretrained_weights`` pattern).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.models import UltralyticsDetectionModel, UltralyticsInstSegModel
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=["cat", "dog"], label_ids=["0", "1"], label_groups=[["cat", "dog"]])


def test_model_accepts_checkpoint_name_for_scratch_training() -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n.pt", pretrained=False, label_info=_label_info())
    assert model.model_name == "yolo26n.pt"
    assert model.pretrained is False


def test_model_allows_yaml_config_for_scratch_training() -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
    assert model.model_name == "yolo26n.yaml"
    assert model.pretrained is False


def test_load_checkpoint_creates_fresh_yolo(tmp_path: Path) -> None:
    """load_checkpoint should create a fresh YOLO instance from the checkpoint."""
    model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
    fake_weights = tmp_path / "weights.pt"
    fake_weights.write_bytes(b"fake")

    mock_yolo = MagicMock()
    with patch("getitune.backend.ultralytics.models.base.YOLO", return_value=mock_yolo) as mock_yolo_cls:
        model.load_checkpoint(fake_weights)

    mock_yolo_cls.assert_called_once_with(str(fake_weights), task="detect")
    assert model._yolo is mock_yolo


def test_build_yolo_from_checkpoint_path_skips_yaml_lookup(caplog: pytest.LogCaptureFixture) -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n.pt", pretrained=False, label_info=_label_info())
    mock_yolo = MagicMock()
    with patch("getitune.backend.ultralytics.models.base.YOLO", return_value=mock_yolo) as mock_yolo_cls:
        yolo = model._build_yolo()

    mock_yolo_cls.assert_called_once_with("yolo26n.pt", task="detect")
    assert yolo is mock_yolo
    # The checkpoint carries its own weights, so ``pretrained=False`` cannot
    # take effect — a warning is logged instead of silently ignoring the flag.
    assert "pretrained=False is ignored" in caplog.text


def test_load_checkpoint_raises_on_missing_file() -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        model.load_checkpoint("/nonexistent/weights.pt")


class TestDataInputParams:
    """Tests for the data_input_params property on UltralyticsModel."""

    def test_returns_data_input_params(self) -> None:
        model = UltralyticsDetectionModel(
            model_name="yolo26n.yaml", pretrained=False, imgsz=640, label_info=_label_info()
        )
        params = model.data_input_params
        assert isinstance(params, DataInputParams)

    def test_input_size_matches_imgsz(self) -> None:
        model = UltralyticsDetectionModel(
            model_name="yolo26n.yaml", pretrained=False, imgsz=320, label_info=_label_info()
        )
        params = model.data_input_params
        assert params.input_size == (320, 320)

    def test_mean_is_zero(self) -> None:
        """YOLO expects identity normalization — mean should be (0, 0, 0)."""
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
        assert model.data_input_params.mean == (0.0, 0.0, 0.0)

    def test_std_is_identity(self) -> None:
        """YOLO uses intensity_config for /255 scaling; std should be identity (1, 1, 1)."""
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
        assert model.data_input_params.std == (1.0, 1.0, 1.0)

    def test_default_imgsz_from_preprocessing_params(self) -> None:
        """When imgsz is not specified, it should come from _default_preprocessing_params."""
        model = UltralyticsDetectionModel(model_name="yolo26n", pretrained=False, label_info=_label_info())
        assert model.imgsz == 640
        assert model.data_input_params.input_size == (640, 640)


class TestExportParameters:
    """Tests for the _export_parameters property on UltralyticsModel."""

    def test_returns_task_level_export_parameters(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        params = model._export_parameters
        assert isinstance(params, TaskLevelExportParameters)

    def test_model_type_is_yolo11(self) -> None:
        """Detection model type should be 'YOLO11' for ModelAPI YOLO adapter."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        assert model._export_parameters.model_type == "YOLO11"

    def test_task_type_is_detection(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        assert model._export_parameters.task_type == "detection"

    def test_model_name_from_model(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
        assert model._export_parameters.model_name == "yolo26n.yaml"

    def test_label_info_from_model(self) -> None:
        li = _label_info()
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=li)
        assert model._export_parameters.label_info == li

    def test_default_thresholds(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        params = model._export_parameters
        # confidence_threshold is None so model_api uses its YOLO11 class default (0.25)
        assert params.confidence_threshold is None
        assert params.iou_threshold == 0.5

    def test_optimization_config_empty(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        assert model._export_parameters.optimization_config == {}

    def test_to_metadata_produces_valid_dict(self) -> None:
        """to_metadata should produce a valid metadata dict with all required keys."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        metadata = model._export_parameters.to_metadata()
        assert ("model_info", "model_type") in metadata
        assert ("model_info", "task_type") in metadata
        assert ("model_info", "labels") in metadata
        assert all(isinstance(v, str) for v in metadata.values())


class TestPretrainedWeights:
    """Tests for the _pretrained_weights pattern."""

    def test_pretrained_weights_defined(self) -> None:
        assert len(UltralyticsDetectionModel._pretrained_weights) > 0

    def test_build_yolo_loads_pretrained_when_enabled(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", pretrained=True, label_info=_label_info())
        mock_yolo = MagicMock()
        with patch("getitune.backend.ultralytics.models.base.YOLO", return_value=mock_yolo):
            yolo = model._build_yolo()

        mock_yolo.load.assert_called_once_with(UltralyticsDetectionModel._pretrained_weights["yolo26n"])
        assert yolo is mock_yolo

    def test_build_yolo_skips_pretrained_when_disabled(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", pretrained=False, label_info=_label_info())
        mock_yolo = MagicMock()
        with patch("getitune.backend.ultralytics.models.base.YOLO", return_value=mock_yolo):
            yolo = model._build_yolo()

        mock_yolo.load.assert_not_called()
        assert yolo is mock_yolo

    def test_detection_contains_yolo26_l_x(self) -> None:
        weights = UltralyticsDetectionModel._pretrained_weights
        assert "yolo26l" in weights
        assert "yolo26x" in weights
        assert "yolo26l.pt" in weights["yolo26l"]
        assert "yolo26x.pt" in weights["yolo26x"]

    @pytest.mark.parametrize("variant", ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"])
    def test_detection_contains_yolo11_variants(self, variant: str) -> None:
        weights = UltralyticsDetectionModel._pretrained_weights
        assert variant in weights
        assert f"{variant}.pt" in weights[variant]

    @pytest.mark.parametrize("variant", ["yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"])
    def test_detection_contains_yolo12_variants(self, variant: str) -> None:
        weights = UltralyticsDetectionModel._pretrained_weights
        assert variant in weights
        assert f"{variant}.pt" in weights[variant]

    def test_detection_all_weights_point_to_v8_4_0(self) -> None:
        """All pretrained weight URLs must reference the v8.4.0 release."""
        for name, url in UltralyticsDetectionModel._pretrained_weights.items():
            assert "v8.4.0" in url, f"URL for {name!r} does not reference v8.4.0: {url}"

    def test_inst_seg_contains_yolo26_l_x(self) -> None:
        weights = UltralyticsInstSegModel._pretrained_weights
        assert "yolo26l-seg" in weights
        assert "yolo26x-seg" in weights
        assert "yolo26l-seg.pt" in weights["yolo26l-seg"]
        assert "yolo26x-seg.pt" in weights["yolo26x-seg"]

    @pytest.mark.parametrize(
        "variant",
        ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"],
    )
    def test_inst_seg_contains_yolo11_variants(self, variant: str) -> None:
        weights = UltralyticsInstSegModel._pretrained_weights
        assert variant in weights
        assert f"{variant}.pt" in weights[variant]

    def test_inst_seg_all_weights_point_to_v8_4_0(self) -> None:
        """All pretrained weight URLs must reference the v8.4.0 release."""
        for name, url in UltralyticsInstSegModel._pretrained_weights.items():
            assert "v8.4.0" in url, f"URL for {name!r} does not reference v8.4.0: {url}"

    def test_detection_preprocessing_params_cover_all_variants(self) -> None:
        """Every entry in _pretrained_weights must have a preprocessing default."""
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
        defaults = model._default_preprocessing_params
        assert isinstance(defaults, dict)
        for variant in UltralyticsDetectionModel._pretrained_weights:
            assert variant in defaults, f"No preprocessing params for detection variant {variant!r}"

    def test_inst_seg_preprocessing_params_cover_all_variants(self) -> None:
        """Every entry in _pretrained_weights must have a preprocessing default."""
        model = UltralyticsInstSegModel(model_name="yolo26n-seg.yaml", pretrained=False, label_info=_label_info())
        defaults = model._default_preprocessing_params
        assert isinstance(defaults, dict)
        for variant in UltralyticsInstSegModel._pretrained_weights:
            assert variant in defaults, f"No preprocessing params for seg variant {variant!r}"
