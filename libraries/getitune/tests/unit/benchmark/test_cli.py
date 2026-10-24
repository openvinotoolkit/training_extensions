# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for getitune.benchmark.cli."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getitune.benchmark.cli import _build_parser, _cmd_provision, _cmd_report, _cmd_run, _parse_key_value_pairs, main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CATALOG_YAML = textwrap.dedent("""\
    version: 1
    datasets:
      - name: ds_a
        script: "scripts/benchmark_datasets/prepare_ds_a.py"
        size_tier: small
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
def catalog_file(tmp_path: Path) -> Path:
    p = tmp_path / "catalog.yaml"
    p.write_text(CATALOG_YAML)
    return p


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST_YAML)
    return p


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


class TestBuildParser:
    def test_provision_subcommand(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["provision", "--catalog", "cat.yaml", "--data-root", "d/"])
        assert args.command == "provision"
        assert args.catalog == Path("cat.yaml")
        assert args.data_root == Path("d/")

    def test_run_subcommand_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"
        assert args.catalog == Path("benchmark_catalog.yaml")
        assert args.manifest == Path("benchmark_manifest.yaml")
        assert args.output_root == Path("results")
        assert args.accelerator == "gpu"
        assert args.deterministic is None
        assert args.dry_run is False
        assert args.max_attempts == 3
        assert args.enable_openvino_benchmark is False
        assert args.enable_validation is True

    def test_run_subcommand_benchmark_options(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--benchmark",
                "--openvino-device",
                "CPU",
                "--training-device-name",
                "NVIDIA GeForce RTX 3090",
                "--openvino-full-device-name",
                "Intel Core i9-14900K",
            ]
        )
        assert args.enable_openvino_benchmark is True
        assert args.openvino_device == "CPU"
        assert args.training_device_name == "NVIDIA GeForce RTX 3090"
        assert args.openvino_device_name == "Intel Core i9-14900K"

    def test_run_subcommand_skip_test(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--skip-test"])
        assert args.enable_validation is False

    def test_run_subcommand_deterministic_flag(self) -> None:
        parser = _build_parser()
        assert parser.parse_args(["run"]).deterministic is None
        assert parser.parse_args(["run", "--deterministic"]).deterministic is True
        assert parser.parse_args(["run", "--no-deterministic"]).deterministic is False

    def test_run_subcommand_filters(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--task",
                "detection",
                "--model",
                "yolox_s",
                "--dataset",
                "ds_a",
                "--priority",
                "core",
                "--size-tier",
                "small",
                "--data-group",
                "weekly",
                "--scenario",
                "default",
                "--scenario-tag",
                "configurable",
                "--num-seeds",
                "5",
                "--max-epochs",
                "10",
                "--eval-upto",
                "export",
                "--dry-run",
            ]
        )
        assert args.task == ["detection"]
        assert args.model == ["yolox_s"]
        assert args.dataset == ["ds_a"]
        assert args.priority == ["core"]
        assert args.size_tier == ["small"]
        assert args.data_group == "weekly"
        assert args.scenario == ["default"]
        assert args.scenario_tag == ["configurable"]
        assert args.num_seeds == 5
        assert args.max_epochs == 10
        assert args.eval_upto == "export"
        assert args.dry_run is True

    def test_run_subcommand_data_group_defaults_to_all(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.data_group == "all"

    def test_run_subcommand_rejects_invalid_data_group(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--data-group", "extend"])

    def test_run_subcommand_requires_data_group_value(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--data-group"])

    def test_run_no_deterministic(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--no-deterministic"])
        assert args.deterministic is False

    def test_run_max_attempts(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--max-attempts", "1"])
        assert args.max_attempts == 1

    def test_log_level_flag(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["provision", "--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_log_level_default_is_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["provision"])
        assert args.log_level is None

    def test_no_subcommand_errors(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# Provision sub-command
# ---------------------------------------------------------------------------


class TestCmdProvision:
    def test_provision_calls_provision_datasets(
        self,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        from getitune.benchmark.catalog import load_catalog

        load_catalog(catalog_file)

        parser = _build_parser()
        args = parser.parse_args(["provision", "--catalog", str(catalog_file), "--data-root", str(tmp_path / "data")])

        with patch("getitune.benchmark.catalog.provision_datasets", return_value={}) as mock_provision:
            rc = _cmd_provision(args)

        assert rc == 0
        mock_provision.assert_called_once()

    def test_provision_no_match_returns_zero(
        self,
        tmp_path: Path,
    ) -> None:
        # Write a catalog with no matching datasets
        cat_file = tmp_path / "empty_catalog.yaml"
        cat_file.write_text("version: 1\ndatasets: []\n")

        parser = _build_parser()
        args = parser.parse_args(
            [
                "provision",
                "--catalog",
                str(cat_file),
                "--data-root",
                str(tmp_path / "data"),
                "--dataset",
                "nonexistent",
            ]
        )
        rc = _cmd_provision(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# Run sub-command
# ---------------------------------------------------------------------------


class TestCmdRun:
    @patch("getitune.benchmark.runner.BenchmarkRunner")
    def test_dry_run_returns_zero(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_runner = MagicMock()
        mock_runner.run.return_value = ([], [])
        mock_runner_cls.return_value = mock_runner

        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--catalog",
                str(catalog_file),
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
                "--dry-run",
            ]
        )
        rc = _cmd_run(args)

        assert rc == 0
        mock_runner.run.assert_called_once()

    @patch("getitune.benchmark.runner.BenchmarkRunner")
    def test_run_with_failures_returns_nonzero(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        from getitune.benchmark.experiment import ExperimentResult

        failure = ExperimentResult(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            seed=0,
            success=False,
            error="boom",
        )
        mock_runner = MagicMock()
        mock_runner.run.return_value = ([], [failure])
        mock_runner_cls.return_value = mock_runner

        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--catalog",
                str(catalog_file),
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
            ]
        )
        rc = _cmd_run(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# Report sub-command
# ---------------------------------------------------------------------------


class TestCmdReport:
    """Regression tests for the ``report`` sub-command including failures.

    Ensures failed MLflow runs (``tags.status = 'failed'``) are surfaced in
    the generated report instead of being silently dropped (see bug where
    ``_cmd_report`` only ever queried successful runs and hardcoded
    ``failures=[]``).
    """

    @patch("getitune.benchmark.report.generate_report")
    @patch("getitune.benchmark.tracking.get_git_sha", return_value="abc123")
    @patch("getitune.benchmark.tracking.get_git_branch", return_value="develop")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_tracking_uri")
    def test_report_includes_failed_runs(
        self,
        _mock_set_uri: MagicMock,
        mock_get_exp: MagicMock,
        _mock_create_exp: MagicMock,
        _mock_set_exp: MagicMock,
        mock_client_cls: MagicMock,
        _mock_git_branch: MagicMock,
        _mock_git_sha: MagicMock,
        mock_generate_report: MagicMock,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"
        mock_get_exp.return_value = mock_experiment  # tracker.setup()'s module-level lookup

        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = mock_experiment  # cli.py's client-level lookup
        mock_client.search_experiments.return_value = []  # no baselines available

        success_run = MagicMock()
        success_run.data.tags = {
            "task": "detection",
            "model": "model_a",
            "dataset": "ds_a",
            "scenario": "default",
            "seed": "0",
        }
        success_run.data.metrics = {"training:val/mAP": 0.5, "duration_seconds": 120.0}

        failed_run = MagicMock()
        failed_run.info.run_id = "run-failed-1"
        failed_run.data.tags = {
            "task": "detection",
            "model": "model_a",
            "dataset": "ds_a",
            "scenario": "default",
            "seed": "1",
            "error": "RuntimeError: could not create a primitive",
            "error_phase": "train",
        }
        failed_run.data.metrics = {}

        def _search_runs(
            *,
            experiment_ids: list[str],
            filter_string: str,
            order_by: list[str],
            max_results: int,
        ) -> list[MagicMock]:
            if "status = 'success'" in filter_string:
                return [success_run]
            if "status = 'failed'" in filter_string:
                return [failed_run]
            return []

        mock_client.search_runs.side_effect = _search_runs

        traceback_path = tmp_path / "traceback.txt"
        traceback_path.write_text("Traceback (most recent call last):\nRuntimeError: could not create a primitive\n")
        mock_client.download_artifacts.return_value = str(traceback_path)

        parser = _build_parser()
        args = parser.parse_args(
            [
                "report",
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
                "--mlflow-uri",
                "http://localhost:5000",
                "--branch",
                "develop",
            ]
        )

        rc = _cmd_report(args)

        assert rc == 0
        mock_generate_report.assert_called_once()
        call_kwargs = mock_generate_report.call_args.kwargs

        results = call_kwargs["results"]
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].model == "model_a"

        failures = call_kwargs["failures"]
        assert len(failures) == 1
        failure = failures[0]
        assert failure.success is False
        assert failure.model == "model_a"
        assert failure.seed == 1
        assert "could not create a primitive" in failure.error
        assert failure.failed_phase == "train"
        assert failure.traceback is not None
        assert "could not create a primitive" in failure.traceback

    @patch("getitune.benchmark.report.generate_report")
    @patch("getitune.benchmark.tracking.get_git_sha", return_value="abc123")
    @patch("getitune.benchmark.tracking.get_git_branch", return_value="develop")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_tracking_uri")
    def test_report_falls_back_to_error_tag_when_traceback_unavailable(
        self,
        _mock_set_uri: MagicMock,
        mock_get_exp: MagicMock,
        _mock_create_exp: MagicMock,
        _mock_set_exp: MagicMock,
        mock_client_cls: MagicMock,
        _mock_git_branch: MagicMock,
        _mock_git_sha: MagicMock,
        mock_generate_report: MagicMock,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        """When the traceback artifact cannot be downloaded (e.g. the MLflow
        server uses an unreachable local-filesystem artifact store), the report
        must still surface the failure by falling back to the short ``error``
        tag rather than dropping the traceback entirely.
        """
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"
        mock_get_exp.return_value = mock_experiment

        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_experiments.return_value = []

        failed_run = MagicMock()
        failed_run.info.run_id = "run-failed-1"
        failed_run.data.tags = {
            "task": "detection",
            "model": "model_a",
            "dataset": "ds_a",
            "scenario": "default",
            "seed": "1",
            "error": "RuntimeError: Masks are required for metric computation",
            "error_phase": "test/export",
        }
        failed_run.data.metrics = {}

        def _search_runs(
            *,
            experiment_ids: list[str],
            filter_string: str,
            order_by: list[str],
            max_results: int,
        ) -> list[MagicMock]:
            if "status = 'failed'" in filter_string:
                return [failed_run]
            return []

        mock_client.search_runs.side_effect = _search_runs

        # Simulate an unreachable artifact store (HTTP 500 / missing file).
        mock_client.download_artifacts.side_effect = RuntimeError("500 Internal Server Error")

        parser = _build_parser()
        args = parser.parse_args(
            [
                "report",
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
                "--mlflow-uri",
                "http://localhost:5000",
                "--branch",
                "develop",
            ]
        )

        rc = _cmd_report(args)

        assert rc == 0
        call_kwargs = mock_generate_report.call_args.kwargs
        failures = call_kwargs["failures"]
        assert len(failures) == 1
        failure = failures[0]
        assert failure.success is False
        assert failure.failed_phase == "test/export"
        # Fallback: traceback is populated from the short error tag.
        assert failure.error == "RuntimeError: Masks are required for metric computation"
        assert failure.traceback == failure.error

    @patch("getitune.benchmark.report.generate_report")
    @patch("getitune.benchmark.tracking.get_git_sha", return_value="abc123")
    @patch("getitune.benchmark.tracking.get_git_branch", return_value="develop")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_tracking_uri")
    def test_report_no_runs_returns_zero(
        self,
        _mock_set_uri: MagicMock,
        mock_get_exp: MagicMock,
        _mock_create_exp: MagicMock,
        _mock_set_exp: MagicMock,
        mock_client_cls: MagicMock,
        _mock_git_branch: MagicMock,
        _mock_git_sha: MagicMock,
        mock_generate_report: MagicMock,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"
        mock_get_exp.return_value = mock_experiment

        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []

        parser = _build_parser()
        args = parser.parse_args(
            [
                "report",
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
                "--mlflow-uri",
                "http://localhost:5000",
                "--branch",
                "develop",
            ]
        )

        rc = _cmd_report(args)

        assert rc == 0
        mock_generate_report.assert_not_called()

    @patch("getitune.benchmark.tracking.get_git_sha", return_value="abc123")
    @patch("getitune.benchmark.tracking.get_git_branch", return_value="develop")
    @patch("mlflow.tracking.MlflowClient")
    @patch("mlflow.set_experiment")
    @patch("mlflow.create_experiment")
    @patch("mlflow.get_experiment_by_name")
    @patch("mlflow.set_tracking_uri")
    def test_report_experiment_not_found_returns_one(
        self,
        _mock_set_uri: MagicMock,
        mock_get_exp: MagicMock,
        _mock_create_exp: MagicMock,
        _mock_set_exp: MagicMock,
        mock_client_cls: MagicMock,
        _mock_git_branch: MagicMock,
        _mock_git_sha: MagicMock,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "1"
        mock_get_exp.return_value = mock_experiment  # tracker.setup() succeeds

        mock_client = mock_client_cls.return_value
        mock_client.get_experiment_by_name.return_value = None  # but the report query fails to find it

        parser = _build_parser()
        args = parser.parse_args(
            [
                "report",
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
                "--mlflow-uri",
                "http://localhost:5000",
                "--branch",
                "develop",
            ]
        )

        rc = _cmd_report(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    @patch("getitune.benchmark.cli.sys.exit")
    @patch("getitune.benchmark.cli._cmd_provision", return_value=0)
    def test_main_dispatches_provision(
        self,
        mock_cmd: MagicMock,
        mock_exit: MagicMock,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        with patch(
            "sys.argv",
            ["prog", "provision", "--catalog", str(catalog_file), "--data-root", str(tmp_path)],
        ):
            main()
        mock_cmd.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch("getitune.benchmark.cli.sys.exit")
    @patch("getitune.benchmark.cli._cmd_run", return_value=0)
    def test_main_dispatches_run(
        self,
        mock_cmd: MagicMock,
        mock_exit: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        with patch(
            "sys.argv",
            [
                "prog",
                "run",
                "--catalog",
                str(catalog_file),
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
            ],
        ):
            main()
        mock_cmd.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch("getitune.benchmark.cli.sys.exit")
    @patch("getitune.benchmark.cli._cmd_run", return_value=1)
    def test_main_propagates_nonzero_exit(
        self,
        mock_cmd: MagicMock,
        mock_exit: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        with patch(
            "sys.argv",
            ["prog", "run", "--catalog", str(catalog_file), "--manifest", str(manifest_file)],
        ):
            main()
        mock_exit.assert_called_once_with(1)


# ---------------------------------------------------------------------------
# CLI filters pass-through to RunConfig
# ---------------------------------------------------------------------------


class TestCmdRunFilters:
    @patch("getitune.benchmark.runner.BenchmarkRunner")
    def test_all_filters_passed_through(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        manifest_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_runner = MagicMock()
        mock_runner.run.return_value = ([], [])
        mock_runner_cls.return_value = mock_runner

        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--catalog",
                str(catalog_file),
                "--manifest",
                str(manifest_file),
                "--output-root",
                str(tmp_path / "results"),
                "--task",
                "detection",
                "--model",
                "yolox_s",
                "--dataset",
                "ds_a",
                "--priority",
                "core",
                "--size-tier",
                "tiny",
                "--scenario",
                "default",
                "--scenario-tag",
                "special",
                "--num-seeds",
                "3",
                "--max-epochs",
                "10",
                "--eval-upto",
                "export",
                "--accelerator",
                "xpu",
            ]
        )
        rc = _cmd_run(args)
        assert rc == 0

        # Verify RunConfig was constructed with correct parameters
        call_kwargs = mock_runner_cls.call_args
        config = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("config")
        if config is None:
            # Constructed positionally
            config = mock_runner_cls.call_args[0][0]
        assert config.accelerator == "xpu"
        assert config.max_epochs == 10
        assert config.num_seeds == 3
        assert config.eval_upto == "export"

    @patch("getitune.benchmark.runner.BenchmarkRunner")
    def test_provision_with_dataset_filter(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        """Provision with --dataset filter should only provision matching datasets."""
        parser = _build_parser()
        args = parser.parse_args(
            [
                "provision",
                "--catalog",
                str(catalog_file),
                "--data-root",
                str(tmp_path / "data"),
                "--dataset",
                "ds_a",
            ]
        )
        with patch("getitune.benchmark.catalog.provision_datasets", return_value={}) as mock_prov:
            rc = _cmd_provision(args)
        assert rc == 0
        mock_prov.assert_called_once()

    @patch("getitune.benchmark.runner.BenchmarkRunner")
    def test_provision_with_size_tier_filter(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "provision",
                "--catalog",
                str(catalog_file),
                "--data-root",
                str(tmp_path / "data"),
                "--size-tier",
                "small",
            ]
        )
        with patch("getitune.benchmark.catalog.provision_datasets", return_value={}) as mock_prov:
            rc = _cmd_provision(args)
        assert rc == 0
        mock_prov.assert_called_once()

    @patch("getitune.benchmark.runner.BenchmarkRunner")
    def test_provision_with_data_group_filter(
        self,
        mock_runner_cls: MagicMock,
        catalog_file: Path,
        tmp_path: Path,
    ) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "provision",
                "--catalog",
                str(catalog_file),
                "--data-root",
                str(tmp_path / "data"),
                "--data-group",
                "weekly",
            ]
        )
        with patch("getitune.benchmark.catalog.provision_datasets", return_value={}) as mock_prov:
            rc = _cmd_provision(args)
        assert rc == 0
        mock_prov.assert_called_once()


# ---------------------------------------------------------------------------
# --override / --train-kwarg / --rotation flags
# ---------------------------------------------------------------------------


class TestParseKeyValuePairs:
    def test_empty_returns_empty(self) -> None:
        assert _parse_key_value_pairs(None) == {}
        assert _parse_key_value_pairs([]) == {}

    def test_single_pair(self) -> None:
        result = _parse_key_value_pairs(["lr=0.01"])
        assert result == {"lr": "0.01"}

    def test_multiple_pairs(self) -> None:
        result = _parse_key_value_pairs(["lr=0.01", "precision=32"])
        assert result == {"lr": "0.01", "precision": "32"}

    def test_dotpath_key(self) -> None:
        result = _parse_key_value_pairs(["model.init_args.optimizer.init_args.lr=0.01"])
        assert result == {"model.init_args.optimizer.init_args.lr": "0.01"}

    def test_invalid_format_raises(self) -> None:
        import argparse as _argparse

        with pytest.raises(_argparse.ArgumentTypeError):
            _parse_key_value_pairs(["no_equals_sign"])


class TestNewCLIFlags:
    def test_override_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--override",
                "model.init_args.optimizer.init_args.lr=0.01",
                "batch_size=16",
            ]
        )
        assert args.override == ["model.init_args.optimizer.init_args.lr=0.01", "batch_size=16"]

    def test_train_kwarg_flag_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(
            [
                "run",
                "--train-kwarg",
                "precision=32",
            ]
        )
        assert args.train_kwarg == ["precision=32"]

    def test_rotation_group_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--rotation-group", "2"])
        assert args.rotation_group == 2

    def test_no_rotation_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "--no-rotation"])
        assert args.no_rotation is True

    def test_defaults_for_new_flags(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run"])
        assert args.override is None
        assert args.train_kwarg is None
        assert args.rotation_group is None
        assert args.no_rotation is False
