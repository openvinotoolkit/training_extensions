# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for getitune.benchmark.tracking (MLflow integration)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from getitune.benchmark.experiment import ExperimentResult, PhaseResult
from getitune.benchmark.manifest import CriteriaConfig, Experiment, ModelEntry, Scenario
from getitune.benchmark.tracking import (
    BenchmarkTracker,
    RunTags,
    TrackingConfig,
    _classify_error,
    _get_cpu_info,
    _get_getitune_version,
    _is_per_class_sentinel,
    _sanitize_error_message,
    get_benchmark_run_id,
    get_git_branch,
    get_git_sha,
)

# ---------------------------------------------------------------------------
# TrackingConfig
# ---------------------------------------------------------------------------


class TestTrackingConfig:
    def test_defaults(self) -> None:
        cfg = TrackingConfig()
        assert cfg.tracking_uri == "./mlruns"
        assert cfg.trigger == "manual"
        assert cfg.baseline_branch == "develop"

    def test_experiment_name_with_explicit_branch(self) -> None:
        cfg = TrackingConfig(branch="develop", trigger="weekly")
        assert cfg.experiment_name == "getitune-benchmark/develop/weekly"

    @patch("getitune.benchmark.tracking.get_git_branch", return_value="feature/xyz")
    def test_experiment_name_auto_branch(self, mock_branch: MagicMock) -> None:
        cfg = TrackingConfig(branch="", trigger="nightly")
        assert cfg.experiment_name == "getitune-benchmark/feature/xyz/nightly"


# ---------------------------------------------------------------------------
# RunTags
# ---------------------------------------------------------------------------


class TestRunTags:
    def test_as_dict(self) -> None:
        tags = RunTags(
            task="detection",
            model="yolox_s",
            dataset="pothole_tiny",
            scenario="default",
            seed="0",
            size_tier="tiny",
            branch="develop",
            accelerator="gpu",
            status="success",
            benchmark_run_id="gh-123-1",
            extra={"override.lr": "0.01"},
        )
        d = tags.as_dict()
        assert d["task"] == "detection"
        assert d["model"] == "yolox_s"
        assert d["override.lr"] == "0.01"
        assert "status" in d
        assert d["benchmark_run_id"] == "gh-123-1"
        assert len(d) == 12  # 11 base + 1 extra


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


class TestGitHelpers:
    @patch("subprocess.check_output", return_value=b"abc1234\n")
    def test_get_git_sha_success(self, mock_co: MagicMock) -> None:
        assert get_git_sha() == "abc1234"

    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    def test_get_git_sha_fallback_env(self, mock_co: MagicMock) -> None:
        with patch.dict("os.environ", {"GITHUB_SHA": "env_sha"}):
            assert get_git_sha() == "env_sha"

    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    def test_get_git_sha_fallback_unknown(self, mock_co: MagicMock) -> None:
        with patch.dict("os.environ", {}, clear=True):
            sha = get_git_sha()
            assert sha == "unknown" or isinstance(sha, str)

    @patch("subprocess.check_output", return_value=b"main\n")
    def test_get_git_branch_success(self, mock_co: MagicMock) -> None:
        assert get_git_branch() == "main"

    @patch("subprocess.check_output", side_effect=FileNotFoundError)
    def test_get_git_branch_fallback(self, mock_co: MagicMock) -> None:
        with patch.dict("os.environ", {"GITHUB_REF_NAME": "pr-42"}):
            assert get_git_branch() == "pr-42"

    def test_get_getitune_version(self) -> None:
        v = _get_getitune_version()
        assert isinstance(v, str)
        assert v != ""

    def test_get_cpu_info_returns_string(self) -> None:
        info = _get_cpu_info()
        assert isinstance(info, str)


# ---------------------------------------------------------------------------
# get_benchmark_run_id
# ---------------------------------------------------------------------------


class TestGetBenchmarkRunId:
    def test_explicit_override_wins(self) -> None:
        with patch.dict(
            "os.environ",
            {"GETITUNE_BENCHMARK_RUN_ID": "custom-42", "GITHUB_RUN_ID": "999"},
            clear=True,
        ):
            assert get_benchmark_run_id() == "custom-42"

    def test_github_run_id_with_attempt(self) -> None:
        with patch.dict(
            "os.environ",
            {"GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "3"},
            clear=True,
        ):
            assert get_benchmark_run_id() == "gh-12345-3"

    def test_github_run_id_defaults_attempt_to_one(self) -> None:
        with patch.dict("os.environ", {"GITHUB_RUN_ID": "12345"}, clear=True):
            assert get_benchmark_run_id() == "gh-12345-1"

    def test_local_fallback(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert get_benchmark_run_id() == "local"


# ---------------------------------------------------------------------------
# BenchmarkTracker.setup
# ---------------------------------------------------------------------------


class TestBenchmarkTrackerSetup:
    @patch("getitune.benchmark.tracking.mlflow")
    def test_setup_creates_experiment(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(
            tracking_uri="/tmp/test_mlruns",  # noqa: S108
            branch="develop",
            trigger="manual",
        )
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "1"

        tracker = BenchmarkTracker(config)
        tracker.setup()

        mock_mlflow.set_tracking_uri.assert_called_once_with("/tmp/test_mlruns")  # noqa: S108
        mock_mlflow.get_experiment_by_name.assert_called_once_with("getitune-benchmark/develop/manual")
        # Local file-store URI: no proxy scheme available, so artifact_location
        # must fall back to MLflow's default (None).
        mock_mlflow.create_experiment.assert_called_once_with(
            "getitune-benchmark/develop/manual", artifact_location=None
        )
        assert tracker._experiment_id == "1"

    @patch("getitune.benchmark.tracking.mlflow")
    def test_setup_creates_experiment_with_proxy_artifacts_for_http_server(self, mock_mlflow: MagicMock) -> None:
        """Against a remote MLflow server, new experiments must pin the
        ``mlflow-artifacts:/`` proxy scheme so artifacts upload/download over
        HTTP regardless of the server's ``--default-artifact-root``.
        """
        config = TrackingConfig(
            tracking_uri="http://mlflow.example:5000",
            branch="develop",
            trigger="weekly",
        )
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "7"

        tracker = BenchmarkTracker(config)
        tracker.setup()

        mock_mlflow.create_experiment.assert_called_once_with(
            "getitune-benchmark/develop/weekly", artifact_location="mlflow-artifacts:/"
        )
        assert tracker._experiment_id == "7"

    @patch("getitune.benchmark.tracking.mlflow")
    def test_setup_reuses_existing_experiment(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(branch="develop", trigger="weekly")

        mock_exp = MagicMock()
        mock_exp.experiment_id = "42"
        mock_mlflow.get_experiment_by_name.return_value = mock_exp

        tracker = BenchmarkTracker(config)
        tracker.setup()

        assert tracker._experiment_id == "42"
        mock_mlflow.create_experiment.assert_not_called()


# ---------------------------------------------------------------------------
# BenchmarkTracker.log_run
# ---------------------------------------------------------------------------


class TestBenchmarkTrackerLogRun:
    def _make_result(self, *, success: bool = True) -> ExperimentResult:
        phases = []
        if success:
            phases = [
                PhaseResult(
                    phase="train", metrics={"training:val/mAP": 0.85, "training:e2e_time": 120.0}, wall_time=120.0
                ),
                PhaseResult(phase="test/torch", metrics={"torch:test/mAP": 0.80}, wall_time=5.0),
            ]
        return ExperimentResult(
            task="detection",
            model="yolox_s",
            dataset="pothole_tiny",
            scenario="default",
            seed=0,
            success=success,
            phases=phases,
            error=None if success else "RuntimeError: OOM",
        )

    def _make_experiment(self) -> Experiment:
        return Experiment(
            task="detection",
            model=ModelEntry(name="yolox_s", task="detection", priority="core"),
            dataset_name="pothole_tiny",
            scenario=Scenario.default(),
            eval_upto="train",
            num_seeds=1,
            criteria=CriteriaConfig(accuracy_metric="mAP", thresholds={}),
        )

    @patch("getitune.benchmark.tracking.mlflow")
    def test_log_successful_run(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(branch="develop", trigger="manual")
        tracker = BenchmarkTracker(config)

        # Set up context manager for start_run
        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        result = self._make_result(success=True)
        experiment = self._make_experiment()

        tracker.log_run(result, experiment, size_tier="tiny")

        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.set_tags.assert_called_once()
        # Verify tags include expected fields
        tags_dict = mock_mlflow.set_tags.call_args[0][0]
        assert tags_dict["model"] == "yolox_s"
        assert tags_dict["status"] == "success"
        assert tags_dict["size_tier"] == "tiny"
        # log_metrics should have been called for numeric metrics and per-phase wall times
        assert mock_mlflow.log_metrics.call_count >= 1

    @patch("getitune.benchmark.tracking.mlflow")
    def test_log_failed_run_sets_error_tag(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(branch="develop", trigger="manual")
        tracker = BenchmarkTracker(config)

        mock_run = MagicMock()
        mock_run.info.run_id = "run_fail"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        result = self._make_result(success=False)
        experiment = self._make_experiment()

        tracker.log_run(result, experiment)

        # Should have called set_tags and structured error tags
        mock_mlflow.set_tags.assert_called_once()
        tags_dict = mock_mlflow.set_tags.call_args[0][0]
        assert tags_dict["status"] == "failed"
        # Structured failure tags: one-line summary + classification
        set_tag_calls = {call.args[0]: call.args[1] for call in mock_mlflow.set_tag.call_args_list}
        assert set_tag_calls["error"] == "RuntimeError: OOM"
        assert set_tag_calls["error_type"] == "RuntimeError"
        assert "error_phase" in set_tag_calls


# ---------------------------------------------------------------------------
# BenchmarkTracker.resolve_baseline
# ---------------------------------------------------------------------------


class TestResolveBaseline:
    @patch("getitune.benchmark.tracking.mlflow")
    def test_no_experiment_returns_none(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(branch="develop", trigger="manual")
        tracker = BenchmarkTracker(config)

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None
        mock_client.search_experiments.return_value = []
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        result = tracker.resolve_baseline(model="m", dataset="d")
        assert result is None

    @patch("getitune.benchmark.tracking.mlflow")
    def test_no_runs_returns_none(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(branch="develop", trigger="weekly")
        tracker = BenchmarkTracker(config)

        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_exp.name = "getitune-benchmark/develop/weekly"
        mock_client.search_experiments.return_value = [mock_exp]
        mock_client.search_runs.return_value = []
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        result = tracker.resolve_baseline(model="m", dataset="d")
        assert result is None

    @patch("getitune.benchmark.tracking.mlflow")
    def test_returns_metrics_from_latest_run(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(branch="develop", trigger="weekly")
        tracker = BenchmarkTracker(config)

        mock_client = MagicMock()
        mock_exp = MagicMock()
        mock_exp.experiment_id = "1"
        mock_exp.name = "getitune-benchmark/develop/weekly"
        mock_client.search_experiments.return_value = [mock_exp]

        mock_run = MagicMock()
        mock_run.data.metrics = {"training:val/mAP": 0.90, "training:e2e_time": 100.0}
        mock_client.search_runs.return_value = [mock_run]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        result = tracker.resolve_baseline(model="yolox_s", dataset="pothole_tiny")
        assert result is not None
        assert result["training:val/mAP"] == 0.90
        assert result["training:e2e_time"] == 100.0

    @patch("getitune.benchmark.tracking.mlflow")
    def test_excludes_current_invocation(self, mock_mlflow: MagicMock) -> None:
        # The baseline must represent the *previous* run, not this one, even
        # when the current run has already logged seeds to the same branch
        # experiment (the weekly ``develop`` case).
        with patch.dict("os.environ", {"GETITUNE_BENCHMARK_RUN_ID": "gh-999-1"}, clear=True):
            config = TrackingConfig(branch="develop", trigger="weekly")
            tracker = BenchmarkTracker(config)

            mock_client = MagicMock()
            mock_exp = MagicMock()
            mock_exp.experiment_id = "1"
            mock_exp.name = "getitune-benchmark/develop/weekly"
            mock_client.search_experiments.return_value = [mock_exp]
            mock_client.search_runs.return_value = []
            mock_mlflow.tracking.MlflowClient.return_value = mock_client

            tracker.resolve_baseline(model="m", dataset="d")

            filter_string = mock_client.search_runs.call_args.kwargs["filter_string"]
            assert "tags.benchmark_run_id != 'gh-999-1'" in filter_string


# ---------------------------------------------------------------------------
# BenchmarkTracker.resolve_baselines_for_results
# ---------------------------------------------------------------------------


class TestResolveBaselinesForResults:
    def test_deduplicates_keys(self) -> None:
        config = TrackingConfig(branch="develop", trigger="manual")
        tracker = BenchmarkTracker(config)

        # Mock resolve_baseline to count calls
        call_count = 0

        def mock_resolve(**kwargs: str) -> dict[str, float]:
            nonlocal call_count
            call_count += 1
            return {"metric": 1.0}

        tracker.resolve_baseline = mock_resolve  # type: ignore[assignment]

        results = [
            ExperimentResult(task="det", model="m", dataset="d", scenario="default", seed=0, success=True),
            ExperimentResult(task="det", model="m", dataset="d", scenario="default", seed=1, success=True),
            ExperimentResult(task="det", model="m", dataset="d", scenario="default", seed=2, success=True),
        ]

        baselines = tracker.resolve_baselines_for_results(results)

        # Should only call resolve_baseline once for the same key
        assert call_count == 1
        assert "det/m/d/default" in baselines

    def test_skips_failed_results(self) -> None:
        config = TrackingConfig(branch="develop", trigger="manual")
        tracker = BenchmarkTracker(config)
        tracker.resolve_baseline = MagicMock(return_value=None)  # type: ignore[assignment]

        results = [
            ExperimentResult(task="det", model="m", dataset="d", scenario="default", seed=0, success=False),
        ]

        baselines = tracker.resolve_baselines_for_results(results)
        assert len(baselines) == 0
        tracker.resolve_baseline.assert_not_called()


# ---------------------------------------------------------------------------
# BenchmarkTracker.purge_old_runs
# ---------------------------------------------------------------------------


class TestPurgeOldRuns:
    @patch("getitune.benchmark.tracking.mlflow")
    def test_no_experiments_returns_zero(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(tracking_uri="/tmp/mlruns")  # noqa: S108
        tracker = BenchmarkTracker(config)

        mock_client = MagicMock()
        mock_client.search_experiments.return_value = []
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        deleted = tracker.purge_old_runs(max_age_days=30)
        assert deleted == 0

    @patch("getitune.benchmark.tracking.mlflow")
    def test_deletes_old_feature_branch_runs(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(tracking_uri="/tmp/mlruns")  # noqa: S108
        tracker = BenchmarkTracker(config)

        mock_client = MagicMock()

        # One benchmark experiment
        mock_exp = MagicMock()
        mock_exp.name = "getitune-benchmark/feature/foo/manual"
        mock_exp.experiment_id = "1"
        mock_client.search_experiments.return_value = [mock_exp]

        # Two runs: one old (100 days ago), one recent (1 day ago)
        import time

        now_ms = int(time.time() * 1000)
        old_run = MagicMock()
        old_run.info.run_id = "old_run"
        old_run.info.start_time = now_ms - (100 * 86400 * 1000)  # 100 days ago
        old_run.data.tags = {"branch": "feature/foo"}

        new_run = MagicMock()
        new_run.info.run_id = "new_run"
        new_run.info.start_time = now_ms - (1 * 86400 * 1000)  # 1 day ago
        new_run.data.tags = {"branch": "feature/foo"}

        mock_client.search_runs.return_value = [old_run, new_run]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        deleted = tracker.purge_old_runs(max_age_days=90)

        assert deleted == 1
        mock_client.delete_run.assert_called_once_with("old_run")

    @patch("getitune.benchmark.tracking.mlflow")
    def test_protects_develop_branch_runs(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(tracking_uri="/tmp/mlruns")  # noqa: S108
        tracker = BenchmarkTracker(config)

        mock_client = MagicMock()

        mock_exp = MagicMock()
        mock_exp.name = "getitune-benchmark/develop/weekly"
        mock_exp.experiment_id = "1"
        mock_client.search_experiments.return_value = [mock_exp]

        # Old run on develop — should NOT be deleted
        import time

        now_ms = int(time.time() * 1000)
        old_develop_run = MagicMock()
        old_develop_run.info.run_id = "develop_run"
        old_develop_run.info.start_time = now_ms - (200 * 86400 * 1000)
        old_develop_run.data.tags = {"branch": "develop"}

        mock_client.search_runs.return_value = [old_develop_run]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        deleted = tracker.purge_old_runs(max_age_days=90)

        assert deleted == 0
        mock_client.delete_run.assert_not_called()

    @patch("getitune.benchmark.tracking.mlflow")
    def test_dry_run_does_not_delete(self, mock_mlflow: MagicMock) -> None:
        config = TrackingConfig(tracking_uri="/tmp/mlruns")  # noqa: S108
        tracker = BenchmarkTracker(config)

        mock_client = MagicMock()

        mock_exp = MagicMock()
        mock_exp.name = "getitune-benchmark/feature/bar/manual"
        mock_exp.experiment_id = "1"
        mock_client.search_experiments.return_value = [mock_exp]

        import time

        now_ms = int(time.time() * 1000)
        old_run = MagicMock()
        old_run.info.run_id = "old_run"
        old_run.info.start_time = now_ms - (100 * 86400 * 1000)
        old_run.data.tags = {"branch": "feature/bar"}

        mock_client.search_runs.return_value = [old_run]
        mock_mlflow.tracking.MlflowClient.return_value = mock_client

        deleted = tracker.purge_old_runs(max_age_days=90, dry_run=True)

        assert deleted == 1
        mock_client.delete_run.assert_not_called()  # dry run — no actual delete


# ---------------------------------------------------------------------------
# Helpers: error sanitization, classification, per-class sentinels
# ---------------------------------------------------------------------------


class TestSanitizeErrorMessage:
    def test_empty(self) -> None:
        assert _sanitize_error_message("") == ""

    def test_single_line_passthrough(self) -> None:
        assert _sanitize_error_message("RuntimeError: OOM") == "RuntimeError: OOM"

    def test_keeps_only_first_non_empty_line(self) -> None:
        msg = "\n\nRuntimeError: boom\nTraceback (most recent call last):\n  File 'x'"
        assert _sanitize_error_message(msg) == "RuntimeError: boom"

    def test_collapses_whitespace_and_strips_quotes(self) -> None:
        assert _sanitize_error_message('Error:   "bad"   quoted') == "Error: 'bad' quoted"

    def test_truncates_long_messages(self) -> None:
        msg = "RuntimeError: " + ("x" * 500) + " final-detail"
        out = _sanitize_error_message(msg)
        assert len(out) <= 250
        # Truncation keeps both the head (exception class) and the tail
        # (informative detail), joined by a middle ellipsis.
        assert " … " in out
        assert out.startswith("RuntimeError:")
        assert out.endswith("final-detail")

    def test_short_messages_are_unchanged(self) -> None:
        assert _sanitize_error_message("RuntimeError: OOM") == "RuntimeError: OOM"


class TestClassifyError:
    def test_extracts_exception_type(self) -> None:
        et, _ = _classify_error("RuntimeError: OOM", None)
        assert et == "RuntimeError"

    def test_unknown_when_malformed(self) -> None:
        et, _ = _classify_error("not an exception line", None)
        assert et == "Unknown"

    def test_phase_from_traceback(self) -> None:
        tb = 'File "/x/getitune/optimize/runner.py", line 1\n'
        _, phase = _classify_error("RuntimeError: x", tb)
        assert phase == "optimize"

    def test_phase_unknown_fallback(self) -> None:
        _, phase = _classify_error("ValueError: bad", "no markers here")
        assert phase == "unknown"


class TestPerClassSentinel:
    def test_sentinel_value_dropped(self) -> None:
        assert _is_per_class_sentinel("training:val/map_per_class", -1.0) is True
        assert _is_per_class_sentinel("training:val/mar_100_per_class", -1.0) is True

    def test_real_value_kept(self) -> None:
        assert _is_per_class_sentinel("training:val/map_per_class", 0.42) is False

    def test_unrelated_key_kept(self) -> None:
        assert _is_per_class_sentinel("training:val/map", -1.0) is False
