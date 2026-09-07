# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics engine."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torchvision import tv_tensors

from getitune.backend.ultralytics.data.geometry import scale_boxes_to_letterbox
from getitune.backend.ultralytics.engine import UltralyticsEngine
from getitune.backend.ultralytics.models import (
    UltralyticsDetectionModel,
    UltralyticsInstSegModel,
    UltralyticsMultiClassClsModel,
)
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import SampleBatch
from getitune.data.module import DataModule
from getitune.types.export import ExportFormat
from getitune.types.label import LabelInfo
from getitune.types.precision import Precision


def _label_info() -> LabelInfo:
    return LabelInfo(
        label_names=["a", "b"],
        label_ids=["0", "1"],
        label_groups=[["a", "b"]],
    )


def _make_engine(tmp_path: Path, mocker) -> tuple[UltralyticsEngine, MagicMock]:
    """Create an engine with a mocked YOLO model for unit testing."""
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

    yolo = MagicMock()
    model._yolo = yolo
    return engine, yolo


def test_train_args_are_train_only(mocker, tmp_path) -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    engine = UltralyticsEngine(
        model=model,
        data=datamodule,
        work_dir=tmp_path,
        device="cpu",
        train_args={"epochs": 7, "lr0": 0.01},
    )

    yolo = MagicMock()
    model._yolo = yolo

    engine.train()
    _, train_kwargs = yolo.train.call_args
    assert train_kwargs["epochs"] == 7
    assert train_kwargs["lr0"] == 0.01
    # Default precision="16-mixed" on CPU is downgraded to FP32 → amp must be False.
    assert train_kwargs["amp"] is False

    with patch.object(engine, "_test_with_datamodule", return_value={}) as test_with_datamodule:
        engine.test()
    test_args, _ = test_with_datamodule.call_args
    assert "epochs" not in test_args[0]
    assert "lr0" not in test_args[0]


def test_ultralytics_engine_supports_ultralytics_model_with_datamodule(mocker) -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    data = mocker.MagicMock(spec=DataModule)

    assert UltralyticsEngine.is_supported(model, data)


def test_engine_propagates_intensity_config_from_datamodule(mocker, tmp_path) -> None:
    """When DataModule has input_intensity_config, engine should propagate it to the model."""
    from getitune.config.data import IntensityConfig

    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    uint16_cfg = IntensityConfig(mode="scale_to_unit", storage_dtype="uint16")
    datamodule.input_intensity_config = uint16_cfg

    engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

    assert model._intensity_config is uint16_cfg
    assert engine._model.data_input_params.intensity_config is uint16_cfg
    assert engine._model.data_input_params.intensity_config.storage_dtype == "uint16"


def test_engine_default_intensity_config_without_datamodule(tmp_path) -> None:
    """When a DataModule with no intensity config is attached, model keeps its default."""
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    datamodule = MagicMock(spec=DataModule)
    datamodule.input_intensity_config = None
    engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path / "work", device="cpu")

    ic = engine._model.data_input_params.intensity_config
    # The model default intensity_config is currently None; the engine only
    # propagates a non-None value from the datamodule.
    assert ic is None


def test_predict_with_datamodule_uses_predict_dataloader(mocker, tmp_path) -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    datamodule.predict_dataloader.return_value = [
        SampleBatch(
            images=tv_tensors.Image(torch.rand(1, 3, 8, 8)),
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(8, 8), ori_shape=(8, 8))  # pyrefly: ignore[no-matching-overload]
            ],
        )
    ]

    engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

    yolo = MagicMock()
    yolo.model = MagicMock()
    yolo.model.to.return_value = yolo.model
    yolo.model.eval.return_value = yolo.model
    yolo.predict.return_value = [
        SimpleNamespace(
            orig_img=np.zeros((8, 8, 3), dtype=np.uint8),
            orig_shape=(8, 8),
            boxes=None,
            masks=None,
        )
    ]
    model._yolo = yolo

    predictions = engine.predict(conf=0.25)

    datamodule.predict_dataloader.assert_called_once_with()
    assert len(predictions) == 1


class TestExportCheckpointResolution:
    """Tests for checkpoint resolution logic in export (now via model.load_checkpoint)."""

    def test_explicit_checkpoint_loads_via_model(self, mocker, tmp_path) -> None:
        """Providing a checkpoint path should call model.load_checkpoint."""
        engine, _ = _make_engine(tmp_path, mocker)

        ckpt_file = tmp_path / "custom.pt"
        ckpt_file.touch()

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml"),
        ):
            engine.export(checkpoint=ckpt_file)

        mock_load.assert_called_once_with(ckpt_file)

    def test_auto_discovers_best_pt(self, mocker, tmp_path) -> None:
        """When no checkpoint given, best.pt from train dir should be used."""
        engine, _ = _make_engine(tmp_path, mocker)

        best_pt = tmp_path / "train" / "weights" / "best.pt"
        best_pt.parent.mkdir(parents=True)
        best_pt.touch()

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml"),
        ):
            engine.export()

        mock_load.assert_called_once_with(best_pt)

    def test_prefers_recorded_train_checkpoint(self, mocker, tmp_path) -> None:
        """Recorded checkpoints from train() should win over hardcoded fallback."""
        engine, _ = _make_engine(tmp_path, mocker)

        recorded_ckpt = tmp_path / "custom_run" / "weights" / "best.pt"
        recorded_ckpt.parent.mkdir(parents=True)
        recorded_ckpt.touch()
        engine._record_last_train_checkpoint(recorded_ckpt)

        # The canonical copy should be at best_checkpoint.pt
        canonical = tmp_path / "best_checkpoint.pt"
        assert canonical.exists()

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml"),
        ):
            engine.export()

        mock_load.assert_called_once_with(canonical.resolve())

    def test_no_checkpoint_no_best_pt_does_not_load(self, mocker, tmp_path) -> None:
        """Without checkpoint or best.pt, load_checkpoint should not be called."""
        engine, yolo = _make_engine(tmp_path, mocker)

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml"),
        ):
            engine.export()

        mock_load.assert_not_called()


class TestExport:
    """Tests for the export() method."""

    def test_export_delegates_to_model_export(self, mocker, tmp_path) -> None:
        """export() should delegate to model.export() with correct args."""
        engine, _ = _make_engine(tmp_path, mocker)

        with (
            patch.object(engine._model, "load_checkpoint"),
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml") as mock_export,
        ):
            # Create a checkpoint so load_checkpoint gets called
            best_pt = tmp_path / "train" / "weights" / "best.pt"
            best_pt.parent.mkdir(parents=True)
            best_pt.touch()

            result = engine.export(
                export_format=ExportFormat.OPENVINO,
                export_precision=Precision.FP32,
            )

        mock_export.assert_called_once_with(
            output_dir=engine._work_dir,
            base_name="exported_model",
            export_format=ExportFormat.OPENVINO,
            precision=Precision.FP32,
            export_args=engine._export_args,
        )
        assert result == tmp_path / "exported_model.xml"

    @pytest.mark.parametrize("export_nms", [False, True])
    def test_export_forwards_nms_option(self, mocker, tmp_path, export_nms) -> None:
        engine, _ = _make_engine(tmp_path, mocker)
        observed_export_nms = []

        def export_model(**_kwargs) -> Path:
            observed_export_nms.append(engine._model.export_nms)
            return tmp_path / "exported_model.xml"

        with patch.object(engine._model, "export", side_effect=export_model):
            engine.export(export_nms=export_nms)

        assert observed_export_nms == [export_nms]
        assert engine._model.export_nms is False

    def test_export_defaults_to_nms_disabled(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)
        observed_export_nms = []

        def export_model(**_kwargs) -> Path:
            observed_export_nms.append(engine._model.export_nms)
            return tmp_path / "exported_model.xml"

        with patch.object(engine._model, "export", side_effect=export_model):
            engine.export()

        assert observed_export_nms == [False]

    def test_export_restores_nms_option_when_export_fails(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)
        engine._model.export_nms = True

        with (
            patch.object(engine._model, "export", side_effect=ValueError("export failed")),
            pytest.raises(ValueError, match="export failed"),
        ):
            engine.export(export_nms=False)

        assert engine._model.export_nms

    def test_export_with_explicit_checkpoint(self, mocker, tmp_path) -> None:
        """Checkpoint arg should be forwarded to model.load_checkpoint."""
        engine, _ = _make_engine(tmp_path, mocker)
        ckpt_file = tmp_path / "custom.pt"
        ckpt_file.touch()

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.onnx"),
        ):
            engine.export(
                checkpoint=ckpt_file,
                export_format=ExportFormat.ONNX,
                export_precision=Precision.FP32,
            )

        mock_load.assert_called_once_with(ckpt_file)

    def test_instance_segmentation_export_succeeds(self, mocker, tmp_path) -> None:
        """Segmentation export should succeed and produce an IR file."""
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

        with patch.object(model, "export", return_value=tmp_path / "exported_model.xml") as mock_export:
            result = engine.export(export_format=ExportFormat.OPENVINO, export_precision=Precision.FP32)

        assert result == tmp_path / "exported_model.xml"
        mock_export.assert_called_once()

    def test_train_records_actual_trainer_checkpoint(self, mocker, tmp_path) -> None:
        """train() should persist the checkpoint at a canonical location."""
        engine, yolo = _make_engine(tmp_path, mocker)

        best_ckpt = tmp_path / "custom_train" / "weights" / "best.pt"
        best_ckpt.parent.mkdir(parents=True)
        best_ckpt.touch()

        trainer = SimpleNamespace(best=best_ckpt, last=tmp_path / "custom_train" / "weights" / "last.pt")
        yolo.trainer = trainer
        yolo.train.return_value = {"fitness": 0.1}
        mocker.patch.object(engine._model, "load_checkpoint")

        result = engine.train(name="custom_train")

        canonical = tmp_path / "best_checkpoint.pt"
        assert result == {"ultralytics/fitness": 0.1}
        assert engine._last_train_checkpoint == canonical.resolve()
        assert (tmp_path / ".last_train_checkpoint").read_text(encoding="utf-8").strip() == str(canonical.resolve())

    def test_test_with_datamodule_loads_explicit_checkpoint(self, mocker, tmp_path) -> None:
        """test(checkpoint=...) should validate the requested checkpoint."""
        engine, _ = _make_engine(tmp_path, mocker)
        ckpt_file = tmp_path / "best.pt"
        ckpt_file.touch()

        validator_cls = MagicMock()
        validator = MagicMock()
        validator.return_value = {"metrics/mAP50(B)": 0.5}
        validator_cls.return_value = validator

        with (
            patch.object(engine, "_make_bound_validator", return_value=validator_cls),
            patch.object(engine._model, "load_checkpoint") as mock_load,
        ):
            metrics = engine.test(checkpoint=ckpt_file)

        mock_load.assert_called_once_with(ckpt_file)
        assert metrics == {"val/map_50": 0.5}

    def test_test_with_data_root_loads_explicit_checkpoint(self, tmp_path) -> None:
        """Filesystem validation should use a fresh YOLO model for explicit checkpoint."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        engine = UltralyticsEngine(model=model, data=tmp_path, work_dir=tmp_path / "work", device="cpu")
        ckpt_file = tmp_path / "best.pt"
        ckpt_file.touch()

        mock_yolo = MagicMock()
        mock_yolo.val.return_value = {"metrics/mAP50(B)": 0.25}

        with (
            patch.object(model, "load_checkpoint") as mock_load,
            patch.object(type(model), "yolo", new_callable=lambda: property(lambda _: mock_yolo)),
        ):
            metrics = engine.test(checkpoint=ckpt_file)

        mock_load.assert_called_once_with(ckpt_file)
        mock_yolo.val.assert_called_once()
        assert metrics == {"val/map_50": 0.25}

    def test_engine_loads_persisted_checkpoint_pointer(self, mocker, tmp_path) -> None:
        """Fresh engine instances should reuse the last recorded training checkpoint."""

        recorded_ckpt = tmp_path / "custom_train" / "weights" / "best.pt"
        recorded_ckpt.parent.mkdir(parents=True)
        recorded_ckpt.touch()
        (tmp_path / ".last_train_checkpoint").write_text(str(recorded_ckpt.resolve()), encoding="utf-8")

        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

        assert engine._last_train_checkpoint == recorded_ckpt.resolve()


class TestModelExporter:
    """Tests for the model's _exporter property."""

    def test_returns_ultralytics_exporter(self, mocker, tmp_path) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())

        from getitune.backend.ultralytics.exporter import UltralyticsModelExporter

        assert isinstance(model._exporter, UltralyticsModelExporter)

    def test_uses_model_data_input_params(self, mocker, tmp_path) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        exporter = model._exporter

        assert exporter.data_input_params.mean == (0.0, 0.0, 0.0)
        assert exporter.data_input_params.std == (1.0, 1.0, 1.0)
        assert exporter.data_input_params.intensity_config is None

    def test_default_yolo_preprocessing_values(self, mocker, tmp_path) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        exporter = model._exporter

        assert exporter.resize_mode == "fit_to_window_letterbox"
        assert exporter.pad_value == 114
        assert exporter.swap_rgb is False


class TestExtractProgressCallback:
    """Tests for _extract_progress_callback in UltralyticsEngine."""

    def test_extracts_progress_from_callback(self) -> None:
        """Should extract fn, min_p, max_p from a callback with matching attrs."""
        cb = SimpleNamespace(_on_progress_update=lambda _p: None, _min_p=10.0, _max_p=80.0)
        fn, min_p, max_p = UltralyticsEngine._extract_progress_callback([cb])
        assert fn is cb._on_progress_update
        assert min_p == 10.0
        assert max_p == 80.0

    def test_returns_none_when_no_callbacks(self) -> None:
        """Should return (None, 0, 100) when callbacks is None."""
        fn, min_p, max_p = UltralyticsEngine._extract_progress_callback(None)
        assert fn is None
        assert min_p == 0.0
        assert max_p == 100.0

    def test_returns_none_when_no_matching_callback(self) -> None:
        """Should return (None, 0, 100) when no callback has _on_progress_update."""
        cb = SimpleNamespace(some_other_attr=True)
        fn, min_p, max_p = UltralyticsEngine._extract_progress_callback([cb])
        assert fn is None
        assert min_p == 0.0
        assert max_p == 100.0


class TestPrecisionToAmp:
    """Tests for UltralyticsEngine._precision_to_amp."""

    @pytest.mark.parametrize("precision", ["16-mixed", "16", "bf16-mixed", "bf16"])
    def test_fp16_variants_return_true_on_cuda(self, precision) -> None:
        assert UltralyticsEngine._precision_to_amp(precision, torch.device("cuda")) is True

    @pytest.mark.parametrize("precision", ["16-mixed", "16", "bf16-mixed", "bf16"])
    def test_fp16_variants_return_true_on_xpu(self, precision) -> None:
        assert UltralyticsEngine._precision_to_amp(precision, torch.device("xpu")) is True

    @pytest.mark.parametrize("precision", ["16-mixed", "16", "bf16-mixed", "bf16"])
    def test_fp16_variants_return_false_on_cpu(self, precision) -> None:
        # CPU does not support AMP; must downgrade to FP32 with a warning.
        assert UltralyticsEngine._precision_to_amp(precision, torch.device("cpu")) is False

    @pytest.mark.parametrize("precision", ["32", "32-true"])
    def test_fp32_variants_return_false_on_all_devices(self, precision) -> None:
        for dev in [torch.device("cpu"), torch.device("cuda"), torch.device("xpu")]:
            assert UltralyticsEngine._precision_to_amp(precision, dev) is False

    def test_none_returns_none(self) -> None:
        for dev in [torch.device("cpu"), torch.device("cuda"), torch.device("xpu")]:
            assert UltralyticsEngine._precision_to_amp(None, dev) is None

    def test_none_precision_does_not_inject_amp_into_train_args(self, mocker, tmp_path) -> None:
        """When precision=None, the 'amp' key must be absent from train kwargs."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")
        engine._device = torch.device("cuda")  # no actual CUDA access needed
        mocker.patch.object(engine, "_compute_best_confidence_threshold", return_value=None)

        yolo = MagicMock()
        model._yolo = yolo

        engine.train(precision=None)
        _, train_kwargs = yolo.train.call_args
        assert "amp" not in train_kwargs

    def test_fp32_precision_injects_amp_false(self, mocker, tmp_path) -> None:
        """precision='32' must inject amp=False regardless of device."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")
        engine._device = torch.device("cuda")
        mocker.patch.object(engine, "_compute_best_confidence_threshold", return_value=None)

        yolo = MagicMock()
        model._yolo = yolo

        engine.train(precision="32")
        _, train_kwargs = yolo.train.call_args
        assert train_kwargs["amp"] is False

    def test_fp16_precision_injects_amp_true_on_cuda(self, mocker, tmp_path) -> None:
        """precision='16-mixed' on CUDA must inject amp=True."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")
        engine._device = torch.device("cuda")
        mocker.patch.object(engine, "_compute_best_confidence_threshold", return_value=None)

        yolo = MagicMock()
        model._yolo = yolo

        engine.train(precision="16-mixed")
        _, train_kwargs = yolo.train.call_args
        assert train_kwargs["amp"] is True


class TestPerClassMetrics:
    """Tests for per-class metric extraction in _translate_metrics."""

    def test_per_class_metrics_extracted(self, mocker, tmp_path) -> None:
        """Per-class precision/recall/mAP50/mAP should appear in translated metrics."""
        engine, _ = _make_engine(tmp_path, mocker)

        results = SimpleNamespace(
            results_dict={
                "metrics/mAP50(B)": 0.75,
                "metrics/mAP50-95(B)": 0.45,
                "metrics/precision(B)": 0.80,
                "metrics/recall(B)": 0.70,
            },
            names={0: "cat", 1: "dog"},
            ap_class_index=[0, 1],
        )
        results.class_result = lambda i: [
            (0.85, 0.75, 0.80, 0.50),
            (0.70, 0.60, 0.65, 0.40),
        ][i]

        metrics = engine._translate_metrics(results)  # pyrefly: ignore[bad-argument-type]

        # Aggregate metrics translated
        assert metrics["val/map_50"] == 0.75
        assert metrics["val/map"] == 0.45

        # Per-class metrics
        assert metrics["val/precision/cat"] == 0.85
        assert metrics["val/recall/cat"] == 0.75
        assert metrics["val/map_50/cat"] == 0.80
        assert metrics["val/map/cat"] == 0.50
        assert metrics["val/precision/dog"] == 0.70
        assert metrics["val/recall/dog"] == 0.60
        assert metrics["val/map_50/dog"] == 0.65
        assert metrics["val/map/dog"] == 0.40

    def test_no_per_class_when_missing_attrs(self, mocker, tmp_path) -> None:
        """Should gracefully skip per-class extraction when results lacks attrs."""
        engine, _ = _make_engine(tmp_path, mocker)

        results = SimpleNamespace(
            results_dict={"metrics/mAP50(B)": 0.75},
        )

        metrics = engine._translate_metrics(results)  # pyrefly: ignore[bad-argument-type]

        assert metrics["val/map_50"] == 0.75
        # No per-class keys
        assert not any("/" in k and k.count("/") >= 2 for k in metrics)


class TestSpeedMetric:
    """Tests for ``val/iter_time`` extraction from Ultralytics' ``speed`` dict."""

    def test_speed_dict_summed_and_converted_to_seconds(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)

        results = SimpleNamespace(
            results_dict={"metrics/mAP50(B)": 0.75},
            speed={"preprocess": 1.0, "inference": 8.0, "loss": 0.0, "postprocess": 1.0},
        )

        metrics = engine._translate_metrics(results)  # pyrefly: ignore[bad-argument-type]

        # 10 ms total -> 0.01 s
        assert metrics["val/iter_time"] == pytest.approx(0.01)

    def test_missing_speed_attr_does_not_add_key(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)

        results = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.75})

        metrics = engine._translate_metrics(results)  # pyrefly: ignore[bad-argument-type]

        assert "val/iter_time" not in metrics

    def test_empty_speed_dict_does_not_add_key(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)

        results = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.75}, speed={})

        metrics = engine._translate_metrics(results)  # pyrefly: ignore[bad-argument-type]

        assert "val/iter_time" not in metrics


class TestSummarizeIterTimes:
    """Tests for ``UltralyticsEngine._summarize_iter_times`` (test/torch phase timing)."""

    def test_empty_list_returns_empty_dict(self) -> None:
        assert UltralyticsEngine._summarize_iter_times([]) == {}

    def test_single_sample_used_directly(self) -> None:
        result = UltralyticsEngine._summarize_iter_times([5.0])
        assert result["test/iter_time"] == pytest.approx(5.0)

    def test_first_sample_excluded_as_warmup(self) -> None:
        result = UltralyticsEngine._summarize_iter_times([10.0, 2.0, 3.0, 4.0])
        # mean of [2.0, 3.0, 4.0] = 3.0
        assert result["test/iter_time"] == pytest.approx(3.0)


class TestTorchmetricsEval:
    """Tests for torchmetrics-based test() evaluation."""

    def test_classification_passes_probabilities_and_labels_to_metric(self, mocker, tmp_path) -> None:
        """Classification torchmetrics must receive class scores and class indices."""
        model = UltralyticsMultiClassClsModel(model_name="yolo26n-cls", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

        yolo = MagicMock()
        model._yolo = yolo
        model.ensure_predict_ready = MagicMock()
        yolo.model = MagicMock()
        yolo.model.to = MagicMock(return_value=yolo.model)
        yolo.model.eval = MagicMock(return_value=yolo.model)
        yolo.predict = MagicMock(
            return_value=[
                SimpleNamespace(probs=SimpleNamespace(data=torch.tensor([0.9, 0.1]))),
                SimpleNamespace(probs=SimpleNamespace(data=torch.tensor([0.2, 0.8]))),
            ]
        )

        batch = SampleBatch(
            images=torch.rand(2, 3, 64, 64),
            labels=[torch.tensor([0]), torch.tensor([1])],
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(64, 64), ori_shape=(64, 64)),  # pyrefly: ignore[no-matching-overload]
                ImageInfo(img_idx=1, img_shape=(64, 64), ori_shape=(64, 64)),  # pyrefly: ignore[no-matching-overload]
            ],
        )
        engine._datamodule.test_dataloader = MagicMock(return_value=[batch])  # pyrefly: ignore[missing-attribute]
        engine._datamodule.label_info = _label_info()  # pyrefly: ignore[missing-attribute]

        metric = MagicMock()
        metric.to = MagicMock(return_value=metric)
        metric.compute = MagicMock(return_value={"accuracy": torch.tensor(1.0)})

        result = engine.test(metric=MagicMock(return_value=metric))

        metric.update.assert_called_once()
        _, update_kwargs = metric.update.call_args
        torch.testing.assert_close(update_kwargs["preds"], torch.tensor([[0.9, 0.1], [0.2, 0.8]]))
        torch.testing.assert_close(update_kwargs["target"], torch.tensor([0, 1]))
        assert result["test/accuracy"] == pytest.approx(1.0)

    def test_test_uses_torchmetrics_when_metric_provided(self, mocker, tmp_path) -> None:
        """test() should use torchmetrics path when metric callable is provided."""
        engine, yolo = _make_engine(tmp_path, mocker)

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = lambda _: 2
        mock_result.boxes.xyxy = torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float32)
        mock_result.boxes.conf = torch.tensor([0.9, 0.7], dtype=torch.float32)
        mock_result.boxes.cls = torch.tensor([0, 1], dtype=torch.float32)

        yolo.predict = MagicMock(return_value=[mock_result])
        yolo.model = MagicMock()
        yolo.model.to = MagicMock(return_value=yolo.model)
        yolo.model.eval = MagicMock(return_value=yolo.model)

        batch = SampleBatch(
            images=torch.rand(1, 3, 64, 64),
            bboxes=[
                tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    torch.tensor([[10.0, 20.0, 30.0, 40.0]]), format="XYXY", canvas_size=(64, 64)
                )
            ],
            labels=[torch.tensor([0])],
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(64, 64), ori_shape=(64, 64))  # pyrefly: ignore[no-matching-overload]
            ],
        )
        engine._datamodule.test_dataloader = MagicMock(return_value=[batch])  # pyrefly: ignore[missing-attribute]
        engine._datamodule.label_info = _label_info()  # pyrefly: ignore[missing-attribute]

        mock_metric = MagicMock()
        mock_metric.to = MagicMock(return_value=mock_metric)
        mock_metric.compute = MagicMock(
            return_value={
                "map": torch.tensor(0.75),
                "map_50": torch.tensor(0.90),
                "map_75": torch.tensor(0.60),
                "mar_1": torch.tensor(0.50),
                "mar_10": torch.tensor(0.65),
                "mar_100": torch.tensor(0.70),
                "classes": torch.tensor([0, 1]),
            }
        )
        metric_callable = MagicMock(return_value=mock_metric)

        result = engine.test(metric=metric_callable)

        metric_callable.assert_called_once()
        mock_metric.update.assert_called_once()
        mock_metric.compute.assert_called_once()

        assert result["test/map"] == pytest.approx(0.75)
        assert result["test/map_50"] == pytest.approx(0.90)
        assert result["test/map_75"] == pytest.approx(0.60)
        assert "test/classes" not in result
        assert "test/iter_time" in result
        assert result["test/iter_time"] >= 0.0

    def test_test_falls_back_to_yolo_without_metric(self, mocker, tmp_path) -> None:
        """test() should use YOLO validator when no metric callable is provided."""
        engine, yolo = _make_engine(tmp_path, mocker)

        validator_instance = MagicMock()
        validator_instance.return_value = SimpleNamespace(
            results_dict={"metrics/mAP50(B)": 0.80},
        )

        with patch.object(engine, "_make_bound_validator", return_value=MagicMock(return_value=validator_instance)):
            validator_instance.return_value = SimpleNamespace(
                results_dict={"metrics/mAP50(B)": 0.80},
            )
            result = engine.test(metric=None)

        assert "val/map_50" in result

    def test_format_torchmetrics_results_skips_non_scalar(self) -> None:
        """Non-scalar tensors and known auxiliary keys should be excluded."""
        results = {
            "map": torch.tensor(0.75),
            "map_50": torch.tensor(0.90),
            "classes": torch.tensor([0, 1, 2]),
            "map_per_class": torch.tensor([0.6, 0.7, 0.8]),
            "mar_100_per_class": torch.tensor([0.5, 0.6, 0.7]),
            "ious": {"bbox": torch.ones(3, 3)},
            "f-measure": torch.tensor(0.85),
        }
        formatted = UltralyticsEngine._format_torchmetrics_results(results)

        assert formatted["test/map"] == pytest.approx(0.75)
        assert formatted["test/map_50"] == pytest.approx(0.90)
        assert formatted["test/f-measure"] == pytest.approx(0.85)
        assert "test/classes" not in formatted
        assert "test/map_per_class" not in formatted
        assert "test/mar_100_per_class" not in formatted
        assert "test/ious" not in formatted


class TestScaleBoxesToLetterbox:
    """Tests for scale_boxes_to_letterbox coordinate transformation."""

    def test_identity_when_image_fits_exactly(self) -> None:
        """When original image matches imgsz, boxes should not change."""
        boxes = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        result = scale_boxes_to_letterbox(boxes, ori_h=640, ori_w=640, imgsz=640)
        assert torch.allclose(result, boxes)

    def test_landscape_image_vertical_padding(self) -> None:
        """480x640 image letterboxed to 640x640: vertical padding of 80px each side."""
        boxes = torch.tensor([[0.0, 0.0, 640.0, 480.0]])
        result = scale_boxes_to_letterbox(boxes, ori_h=480, ori_w=640, imgsz=640)
        # scale = min(640/480, 640/640) = 1.0, pad_x = 0, pad_y = 80
        expected = torch.tensor([[0.0, 80.0, 640.0, 560.0]])
        assert torch.allclose(result, expected)

    def test_portrait_image_horizontal_padding(self) -> None:
        """640x480 image letterboxed to 640x640: horizontal padding of 80px each side."""
        boxes = torch.tensor([[100.0, 50.0, 300.0, 400.0]])
        result = scale_boxes_to_letterbox(boxes, ori_h=640, ori_w=480, imgsz=640)
        # scale = min(640/640, 640/480) = 1.0, pad_x = 80, pad_y = 0
        expected = torch.tensor([[180.0, 50.0, 380.0, 400.0]])
        assert torch.allclose(result, expected)

    def test_small_image_scale_up(self) -> None:
        """320x320 image scaled 2x to fill 640x640: no padding, boxes scaled."""
        boxes = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
        result = scale_boxes_to_letterbox(boxes, ori_h=320, ori_w=320, imgsz=640)
        # scale = 2.0, pad_x = 0, pad_y = 0
        expected = torch.tensor([[20.0, 40.0, 200.0, 400.0]])
        assert torch.allclose(result, expected)

    def test_non_square_with_scaling(self) -> None:
        """960x1280 image scaled 0.5x to 640x640: vertical padding."""
        boxes = torch.tensor([[0.0, 0.0, 1280.0, 960.0]])
        result = scale_boxes_to_letterbox(boxes, ori_h=960, ori_w=1280, imgsz=640)
        # scale = min(640/960, 640/1280) = 0.5, pad_x = 0, pad_y = (640-480)/2 = 80
        expected = torch.tensor([[0.0, 80.0, 640.0, 560.0]])
        assert torch.allclose(result, expected)

    def test_empty_boxes(self) -> None:
        """Empty box tensor should pass through unchanged."""
        boxes = torch.zeros((0, 4))
        result = scale_boxes_to_letterbox(boxes, ori_h=480, ori_w=640, imgsz=640)
        assert result.shape == (0, 4)


class TestInstSegTorchmetrics:
    """Tests for instance segmentation torchmetrics evaluation."""

    def test_test_includes_masks_when_model_produces_them(self, mocker, tmp_path) -> None:
        """test() should include RLE masks in preds/targets for instance seg."""
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

        yolo = MagicMock()
        model._yolo = yolo

        pred_mask = torch.zeros((2, 64, 64))
        pred_mask[0, 10:30, 10:30] = 1.0
        pred_mask[1, 40:60, 40:60] = 1.0

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = lambda _: 2
        mock_result.boxes.xyxy = torch.tensor([[10, 10, 30, 30], [40, 40, 60, 60]], dtype=torch.float32)
        mock_result.boxes.conf = torch.tensor([0.9, 0.7], dtype=torch.float32)
        mock_result.boxes.cls = torch.tensor([0, 1], dtype=torch.float32)
        mock_result.masks = MagicMock()
        mock_result.masks.__len__ = lambda _: 2
        mock_result.masks.data = pred_mask

        yolo.predict = MagicMock(return_value=[mock_result])
        yolo.model = MagicMock()
        yolo.model.to = MagicMock(return_value=yolo.model)
        yolo.model.eval = MagicMock(return_value=yolo.model)

        target_mask = torch.zeros((1, 64, 64), dtype=torch.bool)
        target_mask[0, 10:30, 10:30] = True

        batch = SampleBatch(
            images=torch.rand(1, 3, 64, 64),
            bboxes=[
                tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    torch.tensor([[10.0, 10.0, 30.0, 30.0]]), format="XYXY", canvas_size=(64, 64)
                )
            ],
            labels=[torch.tensor([0])],
            masks=[tv_tensors.Mask(target_mask)],
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(64, 64), ori_shape=(64, 64))  # pyrefly: ignore[no-matching-overload]
            ],
        )
        engine._datamodule.test_dataloader = MagicMock(return_value=[batch])  # pyrefly: ignore[missing-attribute]
        engine._datamodule.label_info = _label_info()  # pyrefly: ignore[missing-attribute]

        mock_metric = MagicMock()
        mock_metric.to = MagicMock(return_value=mock_metric)
        mock_metric.compute = MagicMock(
            return_value={
                "map": torch.tensor(0.60),
                "map_50": torch.tensor(0.80),
            }
        )
        metric_callable = MagicMock(return_value=mock_metric)

        result = engine.test(metric=metric_callable)

        mock_metric.update.assert_called_once()
        call_kwargs = mock_metric.update.call_args
        preds_arg = call_kwargs.kwargs.get("preds") or call_kwargs[1].get("preds") or call_kwargs[0][0]
        targets_arg = call_kwargs.kwargs.get("target") or call_kwargs[1].get("target") or call_kwargs[0][1]

        assert "masks" in preds_arg[0]
        assert len(preds_arg[0]["masks"]) == 2
        assert "counts" in preds_arg[0]["masks"][0]
        assert "size" in preds_arg[0]["masks"][0]

        assert "masks" in targets_arg[0]
        assert len(targets_arg[0]["masks"]) == 1
        assert "counts" in targets_arg[0]["masks"][0]
        assert "size" in targets_arg[0]["masks"][0]

        assert result["test/map"] == pytest.approx(0.60)
        assert result["test/map_50"] == pytest.approx(0.80)

    def test_detection_has_no_masks_key(self, mocker, tmp_path) -> None:
        """Detection-only model should not produce masks key in pred/target dicts."""
        engine, yolo = _make_engine(tmp_path, mocker)

        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.__len__ = lambda _: 1
        mock_result.boxes.xyxy = torch.tensor([[10, 20, 30, 40]], dtype=torch.float32)
        mock_result.boxes.conf = torch.tensor([0.9], dtype=torch.float32)
        mock_result.boxes.cls = torch.tensor([0], dtype=torch.float32)
        mock_result.masks = None

        yolo.predict = MagicMock(return_value=[mock_result])
        yolo.model = MagicMock()
        yolo.model.to = MagicMock(return_value=yolo.model)
        yolo.model.eval = MagicMock(return_value=yolo.model)

        batch = SampleBatch(
            images=torch.rand(1, 3, 64, 64),
            bboxes=[
                tv_tensors.BoundingBoxes(  # pyrefly: ignore[no-matching-overload]
                    torch.tensor([[10.0, 20.0, 30.0, 40.0]]), format="XYXY", canvas_size=(64, 64)
                )
            ],
            labels=[torch.tensor([0])],
            masks=None,
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(64, 64), ori_shape=(64, 64))  # pyrefly: ignore[no-matching-overload]
            ],
        )
        engine._datamodule.test_dataloader = MagicMock(return_value=[batch])  # pyrefly: ignore[missing-attribute]
        engine._datamodule.label_info = _label_info()  # pyrefly: ignore[missing-attribute]

        mock_metric = MagicMock()
        mock_metric.to = MagicMock(return_value=mock_metric)
        mock_metric.compute = MagicMock(return_value={"map": torch.tensor(0.50)})
        metric_callable = MagicMock(return_value=mock_metric)

        engine.test(metric=metric_callable)

        call_kwargs = mock_metric.update.call_args
        preds_arg = call_kwargs.kwargs.get("preds") or call_kwargs[1].get("preds") or call_kwargs[0][0]
        targets_arg = call_kwargs.kwargs.get("target") or call_kwargs[1].get("target") or call_kwargs[0][1]

        assert "masks" not in preds_arg[0]
        assert "masks" not in targets_arg[0]
