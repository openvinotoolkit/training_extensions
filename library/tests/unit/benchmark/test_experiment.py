# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for getitune.benchmark.experiment (result types, resume, metric scraping)."""

from __future__ import annotations

import builtins
import json
import time
from pathlib import Path
from types import ModuleType

import pytest

from getitune.benchmark.experiment import (
    ExperimentExecutor,
    ExperimentResult,
    PhaseExecutionError,
    PhaseResult,
    _find_csv_metrics,
    _get_peak_gpu_memory_mb,
    _PeakRamSampler,
    _recipe_backend,
    _reset_peak_gpu_memory,
    _scrape_csv_metrics,
    _ultralytics_torch_metric,
    _write_phase_metrics_csv,
    detect_resume_point,
    resolve_overrides,
)
from getitune.types.task import TaskType

# ---------------------------------------------------------------------------
# PhaseResult
# ---------------------------------------------------------------------------


class TestPhaseResult:
    def test_defaults(self) -> None:
        r = PhaseResult(phase="train")
        assert r.metrics == {}
        assert r.wall_time == 0.0

    def test_stores_metrics(self) -> None:
        r = PhaseResult(phase="train", metrics={"val/f1": 0.9}, wall_time=42.0)
        assert r.metrics["val/f1"] == 0.9
        assert r.wall_time == 42.0


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:
    def test_all_metrics_empty(self) -> None:
        r = ExperimentResult(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            success=True,
        )
        assert r.all_metrics() == {}

    def test_all_metrics_merges_phases(self) -> None:
        r = ExperimentResult(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            success=True,
            phases=[
                PhaseResult(phase="train", metrics={"a": 1.0}),
                PhaseResult(phase="export", metrics={"b": 2.0}),
            ],
        )
        assert r.all_metrics() == {"a": 1.0, "b": 2.0}

    def test_all_metrics_later_phase_wins(self) -> None:
        """If two phases produce the same key, later phase wins."""
        r = ExperimentResult(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            success=True,
            phases=[
                PhaseResult(phase="train", metrics={"x": 1.0}),
                PhaseResult(phase="test/torch", metrics={"x": 2.0}),
            ],
        )
        assert r.all_metrics()["x"] == 2.0

    def test_failure_factory(self) -> None:
        r = ExperimentResult.failure(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            exc=RuntimeError("OOM"),
        )
        assert r.success is False
        assert r.error is not None
        assert "RuntimeError: OOM" in r.error

    def test_failure_preserves_fields(self) -> None:
        r = ExperimentResult.failure(
            task="seg",
            model="unet",
            dataset="big",
            scenario="tiling",
            seed=3,
            exc=ValueError("bad"),
        )
        assert r.task == "seg"
        assert r.model == "unet"
        assert r.dataset == "big"
        assert r.scenario == "tiling"
        assert r.seed == 3
        assert r.phases == []

    def test_failure_default_phase_is_none(self) -> None:
        r = ExperimentResult.failure(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            exc=RuntimeError("boom"),
        )
        assert r.failed_phase is None

    def test_failure_explicit_phase(self) -> None:
        r = ExperimentResult.failure(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            exc=RuntimeError("boom"),
            failed_phase="export",
        )
        assert r.failed_phase == "export"

    def test_failure_unwraps_phase_execution_error(self) -> None:
        original = RuntimeError("kernel missing")
        r = ExperimentResult.failure(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            exc=PhaseExecutionError("optimize", original),
        )
        # Phase recorded, and error/traceback reflect the *original* exception.
        assert r.failed_phase == "optimize"
        assert r.error == "RuntimeError: kernel missing"
        assert "PhaseExecutionError" not in (r.error or "")

    def test_failure_explicit_phase_overrides_wrapper(self) -> None:
        r = ExperimentResult.failure(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            exc=PhaseExecutionError("optimize", RuntimeError("x")),
            failed_phase="train",
        )
        assert r.failed_phase == "train"


# ---------------------------------------------------------------------------
# PhaseExecutionError
# ---------------------------------------------------------------------------


class TestPhaseExecutionError:
    def test_carries_phase_and_original(self) -> None:
        original = ValueError("bad data")
        err = PhaseExecutionError("train", original)
        assert err.phase == "train"
        assert err.original is original
        assert "train" in str(err)
        assert "bad data" in str(err)


# ---------------------------------------------------------------------------
# resolve_overrides
# ---------------------------------------------------------------------------


class TestResolveOverrides:
    def test_empty(self) -> None:
        assert resolve_overrides({}) == {}

    def test_scalar_passthrough(self) -> None:
        result = resolve_overrides({"lr": 0.01, "epochs": 50, "name": "foo"})
        assert result == {"lr": 0.01, "epochs": 50, "name": "foo"}

    def test_dict_serialized_to_json(self) -> None:
        result = resolve_overrides({"opt": {"class_path": "AdamW", "lr": 0.001}})
        assert json.loads(result["opt"]) == {"class_path": "AdamW", "lr": 0.001}

    def test_list_serialized_to_json(self) -> None:
        result = resolve_overrides({"augmentations": [1, 2, 3]})
        assert json.loads(result["augmentations"]) == [1, 2, 3]

    def test_mixed_types(self) -> None:
        result = resolve_overrides(
            {
                "lr": 0.01,
                "schedule": {"warmup": 5, "decay": 0.1},
                "sizes": [64, 128],
            }
        )
        assert result["lr"] == 0.01
        assert isinstance(result["schedule"], str)
        assert isinstance(result["sizes"], str)


# ---------------------------------------------------------------------------
# detect_resume_point
# ---------------------------------------------------------------------------


class TestDetectResumePoint:
    def test_nonexistent_dir_starts_from_scratch(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "missing"
        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from is None

    def test_empty_dir_starts_from_scratch(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "empty"
        seed_dir.mkdir()
        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from is None

    def test_empty_metrics_csv_starts_from_scratch(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("")
        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from is None
        # Should clean up the directory
        assert not seed_dir.exists()

    def test_metrics_without_checkpoint_is_corrupt(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/f1\n1,0.5\n")
        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from is None
        assert not seed_dir.exists()

    def test_training_done_resumes_from_test_torch(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/f1\n1,0.5\n")
        (seed_dir / "train" / "best_checkpoint.pt").write_text("fake")
        # test/torch marker missing
        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from == "test/torch"

    def test_training_and_test_done_resumes_from_export(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/f1\n1,0.5\n")
        (seed_dir / "train" / "best_checkpoint.pt").write_text("fake")
        (seed_dir / "test" / "torch").mkdir(parents=True)
        (seed_dir / "test" / "torch" / "result.json").write_text("{}")
        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from == "export"

    def test_all_phases_complete_skips(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "seed"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/f1\n1,0.5\n")
        (seed_dir / "train" / "best_checkpoint.pt").write_text("fake")
        (seed_dir / "test" / "torch").mkdir(parents=True)
        (seed_dir / "test" / "torch" / "result.json").write_text("{}")
        (seed_dir / "export").mkdir(parents=True)
        (seed_dir / "export" / "exported_model.xml").write_text("fake")
        (seed_dir / "test" / "export").mkdir(parents=True)
        (seed_dir / "test" / "export" / "result.json").write_text("{}")
        (seed_dir / "optimize").mkdir(parents=True)
        (seed_dir / "optimize" / "optimized_model.xml").write_text("fake")
        (seed_dir / "test" / "optimize").mkdir(parents=True)
        (seed_dir / "test" / "optimize" / "result.json").write_text("{}")

        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is True
        assert resume_from is None


# ---------------------------------------------------------------------------
# _scrape_csv_metrics
# ---------------------------------------------------------------------------


class TestScrapeCsvMetrics:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert _scrape_csv_metrics(tmp_path / "nope.csv", prefix="train:") == {}

    def test_val_metric_takes_max(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("epoch,val/f1\n1,0.3\n2,0.8\n3,0.6\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="training:")
        assert metrics["training:val/f1"] == pytest.approx(0.8)

    def test_epoch_takes_max(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("epoch\n0\n1\n2\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="training:")
        assert metrics["training:epoch"] == 3.0

    def test_iter_time_takes_mean_skipping_first(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("iter_time\n10.0\n2.0\n3.0\n4.0\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="training:")
        # Skips first row (warmup), mean of [2.0, 3.0, 4.0] = 3.0
        assert metrics["training:iter_time"] == pytest.approx(3.0)

    def test_malformed_csv_returns_empty(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("not a csv really\x00\x01\x02")
        # Should not raise, just return empty
        result = _scrape_csv_metrics(csv_path, prefix="x:")
        assert isinstance(result, dict)

    def test_empty_columns_skipped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("val/f1,val/acc\n0.5,\n,0.9\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="p:")
        assert "p:val/f1" in metrics
        assert "p:val/acc" in metrics

    def test_gpu_mem_takes_max(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("gpu_mem\n100\n500\n300\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="training:")
        assert metrics["training:gpu_mem"] == 500.0

    def test_test_metric_is_captured(self, tmp_path: Path) -> None:
        """Regression test: test/* accuracy columns must not be dropped.

        Previously the if/elif chain only recognized "val/", "iter_time",
        "epoch" and "gpu_mem"/"gpu" columns, so test-phase accuracy metrics
        like "test/map" (written by test/torch, test/export, and
        test/optimize phases) fell through silently and never reached the
        benchmark report/CSV/MLflow.
        """
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("test/map,test/f1-score\n0.9,0.75\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="torch:")
        assert metrics["torch:test/map"] == pytest.approx(0.9)
        assert metrics["torch:test/f1-score"] == pytest.approx(0.75)

    def test_test_iter_time_still_averaged(self, tmp_path: Path) -> None:
        """A ``test/iter_time`` column must keep the iter_time averaging behavior."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("test/iter_time\n10.0\n2.0\n4.0\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="torch:")
        assert metrics["torch:test/iter_time"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ExperimentExecutor — construction only (no engine)
# ---------------------------------------------------------------------------


class TestExperimentExecutorInit:
    @pytest.fixture
    def recipe_path(self, tmp_path: Path) -> Path:
        path = tmp_path / "recipe.yaml"
        path.write_text(_LIGHTNING_RECIPE)
        return path

    def test_defaults(self, recipe_path: Path, tmp_path: Path) -> None:
        executor = ExperimentExecutor(
            recipe_path=recipe_path,
            data_path=tmp_path / "data",
            work_dir=tmp_path / "work",
        )
        assert executor.accelerator == "gpu"
        assert executor.seed == 0
        assert executor.deterministic is True
        assert executor.max_epochs is None
        assert executor.scenario_overrides == {}
        assert executor.extra_train_kwargs == {}

    def test_custom_args(self, recipe_path: Path, tmp_path: Path) -> None:
        executor = ExperimentExecutor(
            recipe_path=recipe_path,
            data_path=tmp_path / "data",
            work_dir=tmp_path / "work",
            accelerator="cpu",
            scenario_overrides={"lr": 0.01},
            train_kwargs={"max_epochs": 5},
            seed=42,
            deterministic=False,
            max_epochs=10,
        )
        assert executor.accelerator == "cpu"
        assert executor.seed == 42
        assert executor.deterministic is False
        assert executor.max_epochs == 10
        assert executor.scenario_overrides == {"lr": 0.01}
        assert executor.extra_train_kwargs == {"max_epochs": 5}

    def test_find_exported_model_raises_when_missing(self, recipe_path: Path, tmp_path: Path) -> None:
        executor = ExperimentExecutor(
            recipe_path=recipe_path,
            data_path=tmp_path / "data",
            work_dir=tmp_path / "work",
        )
        with pytest.raises(FileNotFoundError, match="Exported model not found"):
            executor._find_exported_model()

    def test_find_exported_model_primary_path(self, recipe_path: Path, tmp_path: Path) -> None:
        executor = ExperimentExecutor(
            recipe_path=recipe_path,
            data_path=tmp_path / "data",
            work_dir=tmp_path / "work",
        )
        primary = tmp_path / "work" / "export" / "exported_model.xml"
        primary.parent.mkdir(parents=True)
        primary.write_text("<model/>")
        assert executor._find_exported_model() == primary

    def test_find_exported_model_fallback_path(self, recipe_path: Path, tmp_path: Path) -> None:
        executor = ExperimentExecutor(
            recipe_path=recipe_path,
            data_path=tmp_path / "data",
            work_dir=tmp_path / "work",
        )
        fallback = tmp_path / "work" / ".latest" / "export" / "exported_model_decoder.xml"
        fallback.parent.mkdir(parents=True)
        fallback.write_text("<model/>")
        assert executor._find_exported_model() == fallback


# ---------------------------------------------------------------------------
# _get_peak_gpu_memory_mb
# ---------------------------------------------------------------------------


class TestGetPeakGpuMemory:
    def test_returns_float(self) -> None:
        """Should always return a float (0.0 if CUDA unavailable)."""
        result = _get_peak_gpu_memory_mb()
        assert isinstance(result, float)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# _reset_peak_gpu_memory
# ---------------------------------------------------------------------------


class TestResetPeakGpuMemory:
    def test_does_not_raise_when_unavailable(self) -> None:
        """Must be a no-op (never raise) on hosts without CUDA/XPU."""
        _reset_peak_gpu_memory()


# ---------------------------------------------------------------------------
# _PeakRamSampler
# ---------------------------------------------------------------------------


class TestPeakRamSampler:
    def test_peak_mb_reflects_current_process(self) -> None:
        """With psutil installed, the sampler should observe this process' own RSS."""
        pytest.importorskip("psutil")
        with _PeakRamSampler() as sampler:
            # Allocate something so RSS is unambiguously > 0 while sampling.
            _ = bytearray(10 * 1024 * 1024)
            time.sleep(0.3)
        assert sampler.peak_mb > 0.0

    def test_degrades_to_zero_without_psutil(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Falls back to a no-op (peak_mb == 0.0) when psutil can't be imported."""
        real_import = builtins.__import__

        def _raise_for_psutil(name: str, *args, **kwargs) -> ModuleType:
            if name == "psutil":
                msg = "psutil not installed"
                raise ImportError(msg)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _raise_for_psutil)
        with _PeakRamSampler() as sampler:
            time.sleep(0.05)
        assert sampler.peak_mb == 0.0


# ---------------------------------------------------------------------------
# _scrape_csv_metrics — additional edge cases
# ---------------------------------------------------------------------------


class TestScrapeCsvMetricsEdgeCases:
    def test_single_iter_time_row_no_skip(self, tmp_path: Path) -> None:
        """With only one iter_time row, it should still produce a result."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("iter_time\n5.0\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="t:")
        assert "t:iter_time" in metrics

    def test_gpu_column_lowercase_match(self, tmp_path: Path) -> None:
        """Column with 'gpu' (case-insensitive) should take max."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("GPU_utilization\n50\n90\n70\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="t:")
        assert metrics["t:GPU_utilization"] == 90.0

    def test_prefix_applied(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("val/acc\n0.95\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="myprefix:")
        assert "myprefix:val/acc" in metrics

    def test_all_nan_column_skipped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text("val/f1\n\n\n\n")
        metrics = _scrape_csv_metrics(csv_path, prefix="t:")
        assert "t:val/f1" not in metrics

    def test_non_numeric_column_skipped(self, tmp_path: Path) -> None:
        """Non-scalar metrics (e.g. confusion matrices) are skipped, not fatal."""
        csv_path = tmp_path / "metrics.csv"
        csv_path.write_text('test/confusion_matrix,test/accuracy\n"[tensor([[431,   4],\n        [  0,  15]])]",0.95\n')
        metrics = _scrape_csv_metrics(csv_path, prefix="export:")
        assert "export:test/confusion_matrix" not in metrics
        assert metrics["export:test/accuracy"] == 0.95


# ---------------------------------------------------------------------------
# detect_resume_point — additional edge cases
# ---------------------------------------------------------------------------


class TestDetectResumePointEdgeCases:
    def test_resumes_from_optimize(self, tmp_path: Path) -> None:
        """If export + test/export done but optimize missing, resume from optimize."""
        seed_dir = tmp_path / "seed"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/f1\n1,0.5\n")
        (seed_dir / "train" / "best_checkpoint.pt").write_text("fake")
        (seed_dir / "test" / "torch").mkdir(parents=True)
        (seed_dir / "test" / "torch" / "result.json").write_text("{}")
        (seed_dir / "export").mkdir(parents=True)
        (seed_dir / "export" / "exported_model.xml").write_text("fake")
        (seed_dir / "test" / "export").mkdir(parents=True)
        (seed_dir / "test" / "export" / "result.json").write_text("{}")

        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from == "optimize"

    def test_resumes_from_test_optimize(self, tmp_path: Path) -> None:
        """If optimize done but test/optimize missing, resume from test/optimize."""
        seed_dir = tmp_path / "seed"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/f1\n1,0.5\n")
        (seed_dir / "train" / "best_checkpoint.pt").write_text("fake")
        (seed_dir / "test" / "torch").mkdir(parents=True)
        (seed_dir / "test" / "torch" / "result.json").write_text("{}")
        (seed_dir / "export").mkdir(parents=True)
        (seed_dir / "export" / "exported_model.xml").write_text("fake")
        (seed_dir / "test" / "export").mkdir(parents=True)
        (seed_dir / "test" / "export" / "result.json").write_text("{}")
        (seed_dir / "optimize").mkdir(parents=True)
        (seed_dir / "optimize" / "optimized_model.xml").write_text("fake")

        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from == "test/optimize"


# ---------------------------------------------------------------------------
# ExperimentResult — edge cases
# ---------------------------------------------------------------------------


class TestExperimentResultEdgeCases:
    def test_error_field_default_none(self) -> None:
        r = ExperimentResult(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            success=True,
        )
        assert r.error is None

    def test_failure_error_string_format(self) -> None:
        """Various exception types should be captured correctly."""
        r = ExperimentResult.failure(
            task="t",
            model="m",
            dataset="d",
            scenario="s",
            seed=0,
            exc=FileNotFoundError("/path/missing"),
        )
        assert r.error is not None
        assert "FileNotFoundError" in r.error
        assert "/path/missing" in r.error


# ---------------------------------------------------------------------------
# _find_csv_metrics
# ---------------------------------------------------------------------------


class TestFindCsvMetrics:
    def test_finds_version_0(self, tmp_path: Path) -> None:
        """Standard Lightning layout: csv/version_0/metrics.csv."""
        csv_dir = tmp_path / "train"
        csv_file = csv_dir / "csv" / "version_0" / "metrics.csv"
        csv_file.parent.mkdir(parents=True)
        csv_file.write_text("epoch,val/f1\n1,0.5\n")
        found = _find_csv_metrics(csv_dir)
        assert found == csv_file

    def test_finds_latest_version(self, tmp_path: Path) -> None:
        """When multiple version_* dirs exist, pick the highest number."""
        csv_dir = tmp_path / "train"
        for v in [0, 1, 5, 2]:
            p = csv_dir / "csv" / f"version_{v}" / "metrics.csv"
            p.parent.mkdir(parents=True)
            p.write_text(f"epoch\n{v}\n")
        found = _find_csv_metrics(csv_dir)
        assert found is not None
        assert "version_5" in str(found)

    def test_fallback_to_direct_metrics_csv(self, tmp_path: Path) -> None:
        """Fallback when csv/ dir doesn't exist but metrics.csv is directly present."""
        csv_dir = tmp_path / "train"
        csv_dir.mkdir(parents=True)
        direct = csv_dir / "metrics.csv"
        direct.write_text("epoch\n1\n")
        found = _find_csv_metrics(csv_dir)
        assert found == direct

    def test_returns_none_when_nothing_found(self, tmp_path: Path) -> None:
        """No csv dir, no direct metrics.csv → None."""
        csv_dir = tmp_path / "train"
        csv_dir.mkdir(parents=True)
        found = _find_csv_metrics(csv_dir)
        assert found is None

    def test_returns_none_for_nonexistent_dir(self, tmp_path: Path) -> None:
        found = _find_csv_metrics(tmp_path / "nonexistent")
        assert found is None


# ---------------------------------------------------------------------------
# Ultralytics backend support
# ---------------------------------------------------------------------------


_ULTRALYTICS_RECIPE = """\
backend: ultralytics
task: DETECTION
model:
  class_path: getitune.backend.ultralytics.models.detection.UltralyticsDetectionModel
  init_args:
    model_name: yolo26n.yaml
"""

_LIGHTNING_RECIPE = """\
task: DETECTION
model:
  class_path: getitune.backend.lightning.models.detection.atss.ATSS
"""


class TestRecipeBackend:
    def test_ultralytics_recipe_detected(self, tmp_path: Path) -> None:
        recipe = tmp_path / "yolo.yaml"
        recipe.write_text(_ULTRALYTICS_RECIPE)
        backend, task_type = _recipe_backend(recipe)
        assert backend == "ultralytics"
        assert task_type == TaskType.DETECTION

    def test_lightning_recipe_detected(self, tmp_path: Path) -> None:
        recipe = tmp_path / "atss.yaml"
        recipe.write_text(_LIGHTNING_RECIPE)
        backend, task_type = _recipe_backend(recipe)
        assert backend == "lightning"
        assert task_type is None

    def test_missing_recipe_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Could not load recipe"):
            _recipe_backend(tmp_path / "missing.yaml")


class TestUltralyticsTorchMetric:
    def test_detection_has_metric(self) -> None:
        assert _ultralytics_torch_metric(TaskType.DETECTION) is not None

    def test_instance_segmentation_has_metric(self) -> None:
        assert _ultralytics_torch_metric(TaskType.INSTANCE_SEGMENTATION) is not None

    def test_unsupported_task_returns_none(self) -> None:
        assert _ultralytics_torch_metric(TaskType.MULTI_CLASS_CLS) is None
        assert _ultralytics_torch_metric(None) is None


class TestWritePhaseMetricsCsv:
    def test_writes_scalar_metrics(self, tmp_path: Path) -> None:
        _write_phase_metrics_csv(tmp_path, {"test/map": 0.8, "test/f1-score": 0.7})
        csv_file = tmp_path / "csv" / "version_0" / "metrics.csv"
        assert csv_file.exists()
        import pandas as pd

        frame = pd.read_csv(csv_file)
        assert frame["test/map"].iloc[0] == pytest.approx(0.8)

    def test_empty_metrics_no_file(self, tmp_path: Path) -> None:
        _write_phase_metrics_csv(tmp_path, {})
        assert not (tmp_path / "csv").exists()

    def test_non_scalar_metrics_filtered(self, tmp_path: Path) -> None:
        _write_phase_metrics_csv(tmp_path, {"classes": [1, 2, 3]})
        assert not (tmp_path / "csv").exists()


class TestExecutorBackendDispatch:
    def test_lightning_recipe_properties(self, tmp_path: Path) -> None:
        recipe = tmp_path / "atss.yaml"
        recipe.write_text(_LIGHTNING_RECIPE)
        executor = ExperimentExecutor(
            recipe_path=recipe,
            data_path=tmp_path / "data",
            work_dir=tmp_path / "work",
        )
        assert executor.is_ultralytics is False
        assert executor._checkpoint_name == "best_checkpoint.pt"

    def test_ultralytics_recipe_properties(self, tmp_path: Path) -> None:
        recipe = tmp_path / "yolo.yaml"
        recipe.write_text(_ULTRALYTICS_RECIPE)
        executor = ExperimentExecutor(
            recipe_path=recipe,
            data_path=tmp_path / "data",
            work_dir=tmp_path / "work",
        )
        assert executor.is_ultralytics is True
        assert executor._checkpoint_name == "best_checkpoint.pt"


class TestDetectResumePointUltralytics:
    def test_pt_checkpoint_accepted(self, tmp_path: Path) -> None:
        """An Ultralytics ``best_checkpoint.pt`` is a valid train-completion marker."""
        seed_dir = tmp_path / "seed"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/map\n1,0.5\n")
        (seed_dir / "train" / "best_checkpoint.pt").write_text("fake")
        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from == "test/torch"
        assert seed_dir.exists()  # not deleted as corrupt
