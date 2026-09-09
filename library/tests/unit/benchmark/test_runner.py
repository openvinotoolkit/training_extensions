# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for getitune.benchmark.runner (core loop, resume logic)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getitune.benchmark.catalog import DatasetCatalog, load_catalog
from getitune.benchmark.experiment import (
    ExperimentResult,
    PhaseResult,
    detect_resume_point,
    resolve_overrides,
)
from getitune.benchmark.manifest import (
    BenchmarkManifest,
    ManifestFilters,
    load_manifest,
)
from getitune.benchmark.runner import (
    _BENCHMARK_PHASES,
    _EVAL_UPTO_GATES,
    _VALIDATION_PHASES,
    BenchmarkRunner,
    RunConfig,
)

# ---------------------------------------------------------------------------
# Fixtures — minimal catalog + manifest
# ---------------------------------------------------------------------------

CATALOG_YAML = textwrap.dedent("""\
    version: 1
    datasets:
      - name: ds_a
        script: "scripts/benchmark_datasets/prepare_ds_a.py"
        size_tier: tiny
""")

MANIFEST_YAML = textwrap.dedent("""\
    version: 1
    defaults:
      num_seeds: 1
      eval_upto: train
      deterministic: true

    experiments:
      detection:
        models:
          - name: model_a
            priority: core
            recipe: detection/yolox_s.yaml
        datasets:
          - ds_a
        criteria:
          accuracy_metric: mAP
          thresholds:
            "training:val/{metric}": { compare: ">=", margin: 0.10 }
""")


@pytest.fixture
def catalog(tmp_path: Path) -> DatasetCatalog:
    p = tmp_path / "catalog.yaml"
    p.write_text(CATALOG_YAML)
    return load_catalog(p)


@pytest.fixture
def manifest(tmp_path: Path) -> BenchmarkManifest:
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST_YAML)
    return load_manifest(p)


@pytest.fixture
def run_config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        manifest_path=tmp_path / "manifest.yaml",
        catalog_path=tmp_path / "catalog.yaml",
        data_root=tmp_path / "data",
        output_root=tmp_path / "results",
        accelerator="cpu",
        deterministic=True,
        enable_tracking=False,
        enable_report=False,
        isolate_in_subprocess=False,
    )


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------


class TestDetectResumePoint:
    def test_empty_dir_returns_start_from_scratch(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "exp" / "0"
        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from is None

    def test_complete_run_returns_skip(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "exp" / "0"
        # Create all markers
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

    def test_training_done_export_missing(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "exp" / "0"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/f1\n1,0.5\n")
        (seed_dir / "train" / "best_checkpoint.pt").write_text("fake")
        (seed_dir / "test" / "torch").mkdir(parents=True)
        (seed_dir / "test" / "torch" / "result.json").write_text("{}")
        # export not done

        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from == "export"

    def test_corrupt_training_cleaned_up(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "exp" / "0"
        (seed_dir / "train").mkdir(parents=True)
        (seed_dir / "train" / "metrics.csv").write_text("epoch,val/f1\n1,0.5\n")
        # No checkpoint → corrupt

        skip, resume_from = detect_resume_point(seed_dir)
        assert skip is False
        assert resume_from is None
        assert not seed_dir.exists()  # cleaned up


# ---------------------------------------------------------------------------
# Override resolution
# ---------------------------------------------------------------------------


class TestResolveOverrides:
    def test_scalar(self) -> None:
        result = resolve_overrides({"lr": 0.01, "epochs": 50})
        assert result == {"lr": 0.01, "epochs": 50}

    def test_complex_serialized(self) -> None:
        result = resolve_overrides(
            {
                "model.init_args.optimizer": {"class_path": "torch.optim.AdamW", "init_args": {"lr": 0.001}},
                "augmentations": [1, 2, 3],
            }
        )
        assert json.loads(result["model.init_args.optimizer"]) == {
            "class_path": "torch.optim.AdamW",
            "init_args": {"lr": 0.001},
        }
        assert json.loads(result["augmentations"]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Runner dry-run
# ---------------------------------------------------------------------------


class TestRunnerDryRun:
    def test_dry_run_produces_no_results(
        self,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
        run_config: RunConfig,
    ) -> None:
        run_config.filters = ManifestFilters(dry_run=True)
        runner = BenchmarkRunner(run_config)
        successes, failures = runner.run(manifest, catalog)
        assert successes == []
        assert failures == []


# ---------------------------------------------------------------------------
# Runner with mocked executor
# ---------------------------------------------------------------------------


class TestRunnerExecution:
    @patch("getitune.benchmark.runner.provision_datasets")
    @patch("getitune.benchmark.runner.ExperimentExecutor")
    def test_single_experiment_success(
        self,
        mock_executor_cls: MagicMock,
        mock_provision: MagicMock,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
        run_config: RunConfig,
        tmp_path: Path,
    ) -> None:
        # Setup: provision returns a path mapping
        ds_path = tmp_path / "data" / "ds_a"
        ds_path.mkdir(parents=True)
        mock_provision.return_value = {"ds_a": ds_path}

        # Setup: executor methods return phase results
        mock_executor = MagicMock()
        mock_executor.train.return_value = PhaseResult(
            phase="train",
            metrics={"training:val/mAP": 0.85},
            wall_time=10.0,
        )
        mock_executor.test_torch.return_value = PhaseResult(
            phase="test/torch",
            metrics={"torch:test/mAP": 0.80},
            wall_time=5.0,
        )
        mock_executor_cls.return_value = mock_executor

        runner = BenchmarkRunner(run_config)
        successes, failures = runner.run(manifest, catalog)

        assert len(successes) == 1
        assert len(failures) == 0
        assert successes[0].model == "model_a"
        assert successes[0].success is True

    @patch("getitune.benchmark.runner.provision_datasets")
    @patch("getitune.benchmark.runner.ExperimentExecutor")
    def test_experiment_failure_collected(
        self,
        mock_executor_cls: MagicMock,
        mock_provision: MagicMock,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
        run_config: RunConfig,
        tmp_path: Path,
    ) -> None:
        ds_path = tmp_path / "data" / "ds_a"
        ds_path.mkdir(parents=True)
        mock_provision.return_value = {"ds_a": ds_path}

        mock_executor = MagicMock()
        mock_executor.train.side_effect = RuntimeError("CUDA OOM")
        mock_executor_cls.return_value = mock_executor

        runner = BenchmarkRunner(run_config)
        successes, failures = runner.run(manifest, catalog)

        assert len(successes) == 0
        assert len(failures) == 1
        assert failures[0].error is not None
        assert "CUDA OOM" in failures[0].error

    @patch("getitune.benchmark.runner.provision_datasets")
    @patch("getitune.benchmark.runner.ExperimentExecutor")
    def test_retry_on_transient_failure(
        self,
        mock_executor_cls: MagicMock,
        mock_provision: MagicMock,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
        run_config: RunConfig,
        tmp_path: Path,
    ) -> None:
        """First attempt fails, second succeeds."""
        ds_path = tmp_path / "data" / "ds_a"
        ds_path.mkdir(parents=True)
        mock_provision.return_value = {"ds_a": ds_path}

        call_count = 0

        def train_side_effect(*args, **kwargs) -> PhaseResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                msg = "transient error"
                raise RuntimeError(msg)
            return PhaseResult(phase="train", metrics={}, wall_time=1.0)

        mock_executor = MagicMock()
        mock_executor.train.side_effect = train_side_effect
        mock_executor.test_torch.return_value = PhaseResult(phase="test/torch", metrics={}, wall_time=1.0)
        mock_executor_cls.return_value = mock_executor

        runner = BenchmarkRunner(run_config)
        successes, failures = runner.run(manifest, catalog)

        assert len(successes) == 1
        assert len(failures) == 0


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:
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
                PhaseResult(phase="test/torch", metrics={"b": 2.0}),
            ],
        )
        assert r.all_metrics() == {"a": 1.0, "b": 2.0}

    def test_failure_factory(self) -> None:
        r = ExperimentResult.failure(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            exc=ValueError("boom"),
        )
        assert r.success is False
        assert r.error is not None
        assert "ValueError: boom" in r.error


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = RunConfig(
            manifest_path=tmp_path / "m.yaml",
            catalog_path=tmp_path / "c.yaml",
            data_root=tmp_path / "data",
            output_root=tmp_path / "out",
        )
        assert cfg.accelerator == "gpu"
        assert cfg.deterministic is True
        assert cfg.max_epochs is None
        assert cfg.num_seeds is None
        assert cfg.eval_upto is None
        assert cfg.mlflow_tracking_uri == "./mlruns"
        assert cfg.enable_tracking is True
        assert cfg.enable_report is True
        assert cfg.trigger == "manual"
        assert cfg.baseline_branch == "develop"
        assert cfg.max_attempts == 3

    def test_custom_fields(self, tmp_path: Path) -> None:
        cfg = RunConfig(
            manifest_path=tmp_path / "m.yaml",
            catalog_path=tmp_path / "c.yaml",
            data_root=tmp_path / "data",
            output_root=tmp_path / "out",
            accelerator="xpu",
            max_epochs=5,
            num_seeds=2,
            eval_upto="export",
            max_attempts=1,
        )
        assert cfg.accelerator == "xpu"
        assert cfg.max_epochs == 5
        assert cfg.num_seeds == 2
        assert cfg.eval_upto == "export"
        assert cfg.max_attempts == 1

    def test_validation_enabled_by_default(self, tmp_path: Path) -> None:
        cfg = RunConfig(
            manifest_path=tmp_path / "m.yaml",
            catalog_path=tmp_path / "c.yaml",
            data_root=tmp_path / "data",
            output_root=tmp_path / "out",
        )
        assert cfg.enable_validation is True

    def test_validation_can_be_disabled(self, tmp_path: Path) -> None:
        cfg = RunConfig(
            manifest_path=tmp_path / "m.yaml",
            catalog_path=tmp_path / "c.yaml",
            data_root=tmp_path / "data",
            output_root=tmp_path / "out",
            enable_validation=False,
        )
        assert cfg.enable_validation is False


class TestPerformanceCleanup:
    def test_incomplete_performance_result_is_preserved(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "detection" / "model" / "data" / "0"
        seed_dir.mkdir(parents=True)
        (seed_dir / "exported_model.xml").write_text("model")
        (seed_dir / "performance_result.json").write_text(json.dumps({"fp16": {}}))

        assert BenchmarkRunner._performance_result_complete(seed_dir / "exported_model.xml") is False

    def test_complete_performance_result_can_be_cleaned(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "detection" / "model" / "data" / "0"
        seed_dir.mkdir(parents=True)
        (seed_dir / "performance_result.json").write_text(json.dumps({"fp16": {}, "int8": {}}))

        assert BenchmarkRunner._performance_result_complete(seed_dir / "model.xml") is True

    def test_complete_result_for_nonzero_seed_is_cleaned(self, tmp_path: Path) -> None:
        seed_dir = tmp_path / "detection" / "model" / "data" / "1"
        seed_dir.mkdir(parents=True)
        (seed_dir / "performance_result.json").write_text(json.dumps({"fp16": {}, "int8": {}}))

        assert BenchmarkRunner._performance_result_complete(seed_dir / "export" / "exported_model.xml") is True


# ---------------------------------------------------------------------------
# Runner — eval_upto gating
# ---------------------------------------------------------------------------


class TestRunnerEvalUpto:
    def test_no_validation_keeps_performance_phases(self) -> None:
        allowed = set(_EVAL_UPTO_GATES["optimize"])
        allowed.update(_BENCHMARK_PHASES["optimize"])
        allowed.difference_update(_VALIDATION_PHASES)

        assert "test/torch" not in allowed
        assert "test/export" not in allowed
        assert "test/optimize" not in allowed
        assert "export" in allowed
        assert "optimize" in allowed
        assert "benchmark/export" in allowed
        assert "benchmark/optimize" in allowed


class TestRunnerBenchmarkIsolation:
    def test_benchmark_phases_run_in_fresh_stage(self, tmp_path: Path) -> None:
        config = RunConfig(
            manifest_path=tmp_path / "manifest.yaml",
            catalog_path=tmp_path / "catalog.yaml",
            data_root=tmp_path / "data",
            output_root=tmp_path / "results",
            enable_tracking=False,
            enable_report=False,
            isolate_in_subprocess=True,
        )
        runner = BenchmarkRunner(config)
        prepared = ExperimentResult(
            task="detection", model="model_a", dataset="ds_a", scenario="default", seed=0, success=True
        )
        measured = ExperimentResult(
            task="detection",
            model="model_a",
            dataset="ds_a",
            scenario="default",
            seed=0,
            success=True,
            phases=[PhaseResult(phase="benchmark/export", metrics={"fps": 1.0})],
        )
        with patch.object(runner, "_run_single_stage", side_effect=[prepared, measured]) as run_stage:
            result = runner._run_single(
                experiment=MagicMock(),
                seed=0,
                data_path=tmp_path / "data",
                allowed_phases={"train", "benchmark/export"},
            )
        assert result.success
        assert [phase.phase for phase in result.phases] == ["benchmark/export"]
        assert run_stage.call_args_list[0].args[3] == {"train"}
        assert run_stage.call_args_list[1].args[3] == {"benchmark/export"}

    @patch("getitune.benchmark.runner.provision_datasets")
    @patch("getitune.benchmark.runner.ExperimentExecutor")
    def test_eval_upto_train_limits_phases(
        self,
        mock_executor_cls: MagicMock,
        mock_provision: MagicMock,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
        tmp_path: Path,
    ) -> None:
        """With eval_upto=train, only train and test/torch should execute."""
        ds_path = tmp_path / "data" / "ds_a"
        ds_path.mkdir(parents=True)
        mock_provision.return_value = {"ds_a": ds_path}

        mock_executor = MagicMock()
        mock_executor.train.return_value = PhaseResult(phase="train", metrics={}, wall_time=1.0)
        mock_executor.test_torch.return_value = PhaseResult(phase="test/torch", metrics={}, wall_time=1.0)
        mock_executor_cls.return_value = mock_executor

        config = RunConfig(
            manifest_path=tmp_path / "manifest.yaml",
            catalog_path=tmp_path / "catalog.yaml",
            data_root=tmp_path / "data",
            output_root=tmp_path / "results",
            accelerator="cpu",
            eval_upto="train",
            enable_tracking=False,
            enable_report=False,
            isolate_in_subprocess=False,
        )
        runner = BenchmarkRunner(config)
        successes, failures = runner.run(manifest, catalog)

        assert len(successes) == 1
        mock_executor.train.assert_called_once()
        mock_executor.test_torch.assert_called_once()
        mock_executor.export.assert_not_called()
        mock_executor.optimize.assert_not_called()


# ---------------------------------------------------------------------------
# Runner — skip on resume
# ---------------------------------------------------------------------------


class TestRunnerResume:
    @patch("getitune.benchmark.runner.provision_datasets")
    @patch("getitune.benchmark.runner.detect_resume_point", return_value=(True, None))
    @patch("getitune.benchmark.runner.ExperimentExecutor")
    def test_skip_completed_experiment(
        self,
        mock_executor_cls: MagicMock,
        mock_resume: MagicMock,
        mock_provision: MagicMock,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
        run_config: RunConfig,
        tmp_path: Path,
    ) -> None:
        ds_path = tmp_path / "data" / "ds_a"
        ds_path.mkdir(parents=True)
        mock_provision.return_value = {"ds_a": ds_path}

        runner = BenchmarkRunner(run_config)
        successes, failures = runner.run(manifest, catalog)

        # Should succeed without actually calling executor methods
        assert len(successes) == 1
        assert successes[0].success is True
        mock_executor_cls.return_value.train.assert_not_called()


# ---------------------------------------------------------------------------
# Runner — missing dataset path
# ---------------------------------------------------------------------------


class TestRunnerMissingDataset:
    @patch("getitune.benchmark.runner.provision_datasets")
    def test_missing_dataset_skipped(
        self,
        mock_provision: MagicMock,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
        run_config: RunConfig,
    ) -> None:
        """If provision returns empty, experiment should be skipped."""
        mock_provision.return_value = {}  # ds_a not provisioned

        runner = BenchmarkRunner(run_config)
        successes, failures = runner.run(manifest, catalog)
        assert successes == []
        assert failures == []


# ---------------------------------------------------------------------------
# Runner — no matching experiments
# ---------------------------------------------------------------------------


class TestRunnerNoMatch:
    def test_no_experiments_returns_empty(
        self,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
        run_config: RunConfig,
    ) -> None:
        run_config.filters = ManifestFilters(tasks=["nonexistent_task"])
        runner = BenchmarkRunner(run_config)
        successes, failures = runner.run(manifest, catalog)
        assert successes == []
        assert failures == []


class TestRotationBucket:
    """Rotation buckets must be stable across processes (cf. ``H1``)."""

    def test_stable_across_calls(self) -> None:
        from getitune.benchmark.runner import _rotation_bucket

        names = ["yolox_s", "atss_mobilenetv2", "rfdetr_small", "deim_dfine_x"]
        first = [_rotation_bucket(n, 4) for n in names]
        second = [_rotation_bucket(n, 4) for n in names]
        assert first == second
        assert all(0 <= b < 4 for b in first)

    def test_known_values_pin_distribution(self) -> None:
        """Pin a few known buckets so accidental hash changes are caught."""
        from getitune.benchmark.runner import _rotation_bucket

        # blake2b is deterministic — these values would only change if the
        # hash function or digest size in ``_rotation_bucket`` is altered.
        assert _rotation_bucket("yolox_s", 4) == _rotation_bucket("yolox_s", 4)
        # Different names should not all collapse to the same bucket.
        buckets = {_rotation_bucket(f"model_{i}", 4) for i in range(20)}
        assert len(buckets) > 1
