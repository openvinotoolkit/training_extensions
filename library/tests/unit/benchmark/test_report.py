# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for getitune.benchmark.report (regression detection and Markdown generation)."""

from __future__ import annotations

from pathlib import Path

from getitune.benchmark.experiment import ExperimentResult, PhaseResult
from getitune.benchmark.manifest import CriteriaConfig, Threshold
from getitune.benchmark.report import (
    BenchmarkReport,
    ExperimentComparison,
    FailureRecord,
    RegressionResult,
    aggregate_metrics_across_seeds,
    build_report,
    check_regressions,
    generate_csv,
    generate_markdown,
    generate_report,
    write_failures_json,
)

# ---------------------------------------------------------------------------
# RegressionResult
# ---------------------------------------------------------------------------


class TestRegressionResult:
    def test_delta_positive(self) -> None:
        r = RegressionResult(
            metric="val/mAP",
            current_value=0.90,
            baseline_value=0.80,
            margin=0.10,
            direction="higher_is_better",
            status="improvement",
        )
        assert r.delta is not None
        assert abs(r.delta - 0.125) < 1e-6
        assert r.delta_pct == "+12.5%"

    def test_delta_negative(self) -> None:
        r = RegressionResult(
            metric="val/mAP",
            current_value=0.70,
            baseline_value=0.80,
            margin=0.10,
            direction="higher_is_better",
            status="regression",
        )
        assert r.delta is not None
        assert r.delta < 0
        assert r.delta_pct == "-12.5%"

    def test_delta_no_baseline(self) -> None:
        r = RegressionResult(
            metric="val/mAP",
            current_value=0.85,
            baseline_value=None,
            margin=0.10,
            direction="higher_is_better",
            status="no_baseline",
        )
        assert r.delta is None
        assert r.delta_pct == "N/A"

    def test_delta_zero_baseline(self) -> None:
        r = RegressionResult(
            metric="val/mAP",
            current_value=0.5,
            baseline_value=0.0,
            margin=0.10,
            direction="higher_is_better",
            status="pass",
        )
        assert r.delta is None


# ---------------------------------------------------------------------------
# check_regressions
# ---------------------------------------------------------------------------


class TestCheckRegressions:
    def _thresholds(self) -> dict[str, Threshold]:
        return {
            "training:val/mAP": Threshold(compare=">=", margin=0.10),
            "training:e2e_time": Threshold(compare="<=", margin=0.10),
        }

    def test_pass_when_within_margin(self) -> None:
        current = {"training:val/mAP": 0.85, "time/training/e2e": 105.0}
        baseline = {"training:val/mAP": 0.90, "time/training/e2e": 100.0}
        results = check_regressions(current, baseline, self._thresholds())

        status_map = {r.metric: r.status for r in results}
        assert status_map["training:val/mAP"] == "pass"
        assert status_map["training:e2e_time"] == "pass"

    def test_regression_higher_is_better(self) -> None:
        """Accuracy dropped below baseline * (1 - margin)."""
        current = {"training:val/mAP": 0.75}  # 0.75 < 0.90 * 0.90 = 0.81
        baseline = {"training:val/mAP": 0.90}
        results = check_regressions(current, baseline, self._thresholds())

        assert len(results) == 1
        assert results[0].status == "regression"

    def test_regression_lower_is_better(self) -> None:
        """Time increased above baseline * (1 + margin)."""
        current = {"time/training/e2e": 120.0}  # 120 > 100 * 1.10 = 110
        baseline = {"time/training/e2e": 100.0}
        results = check_regressions(current, baseline, self._thresholds())

        time_result = [r for r in results if r.metric == "training:e2e_time"]
        assert len(time_result) == 1
        assert time_result[0].status == "regression"

    def test_timing_threshold_resolves_against_rewritten_baseline(self) -> None:
        """Timing thresholds must successfully match MLflow-rewritten baseline keys."""
        current = {"time/training/e2e": 95.0}
        baseline = {"time/training/e2e": 100.0}
        thresholds = {"training:e2e_time": Threshold(compare="<=", margin=0.10)}
        results = check_regressions(current, baseline, thresholds)

        assert len(results) == 1
        assert results[0].status == "pass"
        assert results[0].baseline_value == 100.0

    def test_improvement_detected(self) -> None:
        """Accuracy improved significantly above baseline * (1 + margin)."""
        current = {"training:val/mAP": 1.05}  # > 0.90 * 1.10 = 0.99
        baseline = {"training:val/mAP": 0.90}
        results = check_regressions(current, baseline, self._thresholds())

        assert results[0].status == "improvement"

    def test_no_baseline_status(self) -> None:
        current = {"training:val/mAP": 0.85}
        results = check_regressions(current, None, self._thresholds())

        assert len(results) == 1
        assert results[0].status == "no_baseline"

    def test_missing_metric_in_baseline(self) -> None:
        current = {"training:val/mAP": 0.85}
        baseline = {}  # metric not in baseline
        results = check_regressions(current, baseline, self._thresholds())

        map_result = [r for r in results if r.metric == "training:val/mAP"]
        assert len(map_result) == 1
        assert map_result[0].status == "no_baseline"

    def test_missing_metric_in_current_is_skipped(self) -> None:
        current = {}  # no metrics
        baseline = {"training:val/mAP": 0.90}
        results = check_regressions(current, baseline, self._thresholds())
        assert len(results) == 0

    def test_empty_thresholds(self) -> None:
        results = check_regressions({"a": 1.0}, {"a": 1.0}, {})
        assert results == []


# ---------------------------------------------------------------------------
# aggregate_metrics_across_seeds
# ---------------------------------------------------------------------------


class TestAggregateMetrics:
    def test_averages_across_seeds(self) -> None:
        results = [
            ExperimentResult(
                task="det",
                model="m",
                dataset="d",
                scenario="default",
                seed=0,
                success=True,
                phases=[PhaseResult(phase="train", metrics={"acc": 0.8}, wall_time=10.0)],
            ),
            ExperimentResult(
                task="det",
                model="m",
                dataset="d",
                scenario="default",
                seed=1,
                success=True,
                phases=[PhaseResult(phase="train", metrics={"acc": 0.9}, wall_time=12.0)],
            ),
        ]

        averaged = aggregate_metrics_across_seeds(results)
        key = ("det", "m", "d", "default")
        assert key in averaged
        assert abs(averaged[key]["acc"] - 0.85) < 1e-6

    def test_skips_failures(self) -> None:
        results = [
            ExperimentResult(task="det", model="m", dataset="d", scenario="default", seed=0, success=False),
        ]
        averaged = aggregate_metrics_across_seeds(results)
        assert len(averaged) == 0

    def test_multiple_experiments(self) -> None:
        results = [
            ExperimentResult(
                task="det",
                model="m1",
                dataset="d",
                scenario="default",
                seed=0,
                success=True,
                phases=[PhaseResult(phase="train", metrics={"acc": 0.8})],
            ),
            ExperimentResult(
                task="det",
                model="m2",
                dataset="d",
                scenario="default",
                seed=0,
                success=True,
                phases=[PhaseResult(phase="train", metrics={"acc": 0.9})],
            ),
        ]
        averaged = aggregate_metrics_across_seeds(results)
        assert ("det", "m1", "d", "default") in averaged
        assert ("det", "m2", "d", "default") in averaged


# ---------------------------------------------------------------------------
# ExperimentComparison
# ---------------------------------------------------------------------------


class TestExperimentComparison:
    def test_no_baseline(self) -> None:
        comp = ExperimentComparison(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            current_metrics={"acc": 0.85},
            baseline_metrics=None,
        )
        assert not comp.has_baseline
        assert comp.overall_status == "no_baseline"
        assert comp.status_emoji == "🆕"

    def test_pass(self) -> None:
        comp = ExperimentComparison(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            current_metrics={"acc": 0.85},
            baseline_metrics={"acc": 0.84},
            regressions=[
                RegressionResult("acc", 0.85, 0.84, 0.10, "higher_is_better", "pass"),
            ],
        )
        assert comp.overall_status == "pass"
        assert comp.status_emoji == "✅"

    def test_regression(self) -> None:
        comp = ExperimentComparison(
            task="det",
            model="m",
            dataset="d",
            scenario="default",
            current_metrics={"acc": 0.50},
            baseline_metrics={"acc": 0.85},
            regressions=[
                RegressionResult("acc", 0.50, 0.85, 0.10, "higher_is_better", "regression"),
            ],
        )
        assert comp.has_regression
        assert comp.overall_status == "regression"
        assert comp.status_emoji == "⚠️"

    def test_key(self) -> None:
        comp = ExperimentComparison(
            task="det",
            model="yolox_s",
            dataset="pothole_tiny",
            scenario="lr_high",
            current_metrics={},
            baseline_metrics=None,
        )
        assert comp.key == "det/yolox_s/pothole_tiny/lr_high"


# ---------------------------------------------------------------------------
# BenchmarkReport
# ---------------------------------------------------------------------------


class TestBenchmarkReport:
    def test_summary_counts(self) -> None:
        report = BenchmarkReport(
            comparisons=[
                ExperimentComparison(
                    task="det",
                    model="m1",
                    dataset="d",
                    scenario="default",
                    current_metrics={},
                    baseline_metrics={"a": 1.0},
                    regressions=[RegressionResult("a", 1.0, 1.0, 0.1, "higher_is_better", "pass")],
                ),
                ExperimentComparison(
                    task="det",
                    model="m2",
                    dataset="d",
                    scenario="default",
                    current_metrics={},
                    baseline_metrics=None,
                ),
            ],
            failures=[
                FailureRecord(task="det", model="m3", dataset="d", scenario="default", seed=0, error="OOM"),
            ],
        )
        assert report.pass_count == 1
        assert report.no_baseline_count == 1
        assert report.regression_count == 0
        assert len(report.failures) == 1
        assert report.total_experiments == 3


# ---------------------------------------------------------------------------
# build_report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_builds_from_results(self) -> None:
        results = [
            ExperimentResult(
                task="det",
                model="m",
                dataset="d",
                scenario="default",
                seed=0,
                success=True,
                phases=[PhaseResult(phase="train", metrics={"training:val/mAP": 0.85})],
            ),
        ]
        failures = [
            ExperimentResult.failure(
                task="det",
                model="m2",
                dataset="d",
                scenario="default",
                seed=0,
                exc=RuntimeError("boom"),
            ),
        ]
        baselines: dict[str, dict[str, float] | None] = {"det/m/d/default": {"training:val/mAP": 0.80}}
        criteria_by_task = {
            "det": CriteriaConfig(
                accuracy_metric="mAP",
                thresholds={"training:val/mAP": Threshold(compare=">=", margin=0.10)},
            ),
        }

        report = build_report(
            results=results,
            failures=failures,
            baselines=baselines,
            criteria_by_task=criteria_by_task,
            branch="develop",
            git_sha="abc",
        )

        assert len(report.comparisons) == 1
        assert len(report.failures) == 1
        assert report.branch == "develop"
        assert report.comparisons[0].model == "m"
        assert report.failures[0].model == "m2"

    def test_task_containing_slash_is_not_misparsed(self) -> None:
        """Regression test: task identifiers like ``classification/multi_class_cls``
        contain a ``/`` themselves. Previously ``build_report`` reconstructed
        task/model/dataset/scenario by splitting a ``"/"``-joined string, which
        shifted every field by one and merged the dataset and scenario into a
        single value whenever the task contained a slash.
        """
        results = [
            ExperimentResult(
                task="classification/multi_class_cls",
                model="dino_v2",
                dataset="flowers102",
                scenario="default",
                seed=0,
                success=True,
                phases=[PhaseResult(phase="train", metrics={"training:val/accuracy": 0.96})],
            ),
        ]
        criteria_by_task = {
            "classification/multi_class_cls": CriteriaConfig(
                accuracy_metric="accuracy",
                thresholds={},
            ),
        }

        report = build_report(
            results=results,
            failures=[],
            baselines={},
            criteria_by_task=criteria_by_task,
        )

        assert len(report.comparisons) == 1
        comp = report.comparisons[0]
        assert comp.task == "classification/multi_class_cls"
        assert comp.model == "dino_v2"
        assert comp.dataset == "flowers102"
        assert comp.scenario == "default"


# ---------------------------------------------------------------------------
# generate_markdown
# ---------------------------------------------------------------------------


class TestGenerateMarkdown:
    def test_contains_header(self) -> None:
        report = BenchmarkReport(comparisons=[], failures=[], branch="develop", git_sha="abc")
        md = generate_markdown(report)
        assert "GetiTune Benchmark Report" in md
        assert "`develop`" in md
        assert "`abc`" in md

    def test_contains_task_table(self) -> None:
        report = BenchmarkReport(
            comparisons=[
                ExperimentComparison(
                    task="detection",
                    model="yolox_s",
                    dataset="pothole",
                    scenario="default",
                    current_metrics={"training:val/mAP": 0.85},
                    baseline_metrics=None,
                ),
            ],
            failures=[],
        )
        md = generate_markdown(report)
        assert "### detection" in md
        assert "yolox_s" in md
        assert "pothole" in md

    def test_contains_failure_section(self) -> None:
        report = BenchmarkReport(
            comparisons=[],
            failures=[
                FailureRecord(task="det", model="m", dataset="d", scenario="default", seed=0, error="OOM"),
            ],
        )
        md = generate_markdown(report)
        assert "Failures" in md
        assert "OOM" in md

    def test_failure_section_shows_stage(self) -> None:
        report = BenchmarkReport(
            comparisons=[],
            failures=[
                FailureRecord(
                    task="det",
                    model="m",
                    dataset="d",
                    scenario="default",
                    seed=0,
                    error="OOM",
                    failed_phase="export",
                ),
            ],
        )
        md = generate_markdown(report)
        assert "Stage" in md
        assert "export" in md

    def test_failure_section_unknown_stage(self) -> None:
        report = BenchmarkReport(
            comparisons=[],
            failures=[
                FailureRecord(task="det", model="m", dataset="d", scenario="default", seed=0, error="OOM"),
            ],
        )
        md = generate_markdown(report)
        # Missing phase falls back to a readable placeholder rather than blank.
        assert "unknown" in md

    def test_contains_regression_alerts(self) -> None:
        report = BenchmarkReport(
            comparisons=[
                ExperimentComparison(
                    task="det",
                    model="m",
                    dataset="d",
                    scenario="default",
                    current_metrics={"val/mAP": 0.50},
                    baseline_metrics={"val/mAP": 0.90},
                    regressions=[
                        RegressionResult("val/mAP", 0.50, 0.90, 0.10, "higher_is_better", "regression"),
                    ],
                ),
            ],
            failures=[],
        )
        md = generate_markdown(report)
        assert "Regression Alerts" in md
        assert "regression" in md.lower()

    def test_empty_report(self) -> None:
        report = BenchmarkReport(comparisons=[], failures=[])
        md = generate_markdown(report)
        assert "GetiTune Benchmark Report" in md
        assert "**0** passed" in md

    def test_test_metric_columns_shown_for_all_backends(self) -> None:
        """torch/export/optimize test accuracy metrics must all render as columns.

        Regression test: the ``optimize:`` bucket was previously missing from
        ``_detect_metric_columns``, so post-quantization test accuracy never
        appeared in the Markdown table even when present in current_metrics.
        """
        report = BenchmarkReport(
            comparisons=[
                ExperimentComparison(
                    task="detection",
                    model="yolox_s",
                    dataset="pothole",
                    scenario="default",
                    current_metrics={
                        "training:val/mAP": 0.85,
                        "torch:test/map": 0.80,
                        "export:test/map": 0.79,
                        "optimize:test/map": 0.77,
                    },
                    baseline_metrics=None,
                ),
            ],
            failures=[],
        )
        md = generate_markdown(report)
        assert "Test map" in md
        assert "Export map" in md
        assert "Optimize map" in md

    def test_peak_ram_column_shown_when_present(self) -> None:
        report = BenchmarkReport(
            comparisons=[
                ExperimentComparison(
                    task="detection",
                    model="yolox_s",
                    dataset="pothole",
                    scenario="default",
                    current_metrics={
                        "training:val/mAP": 0.85,
                        "training:gpu_mem": 512.0,
                        "training:ram_mem": 2048.0,
                    },
                    baseline_metrics=None,
                ),
            ],
            failures=[],
        )
        md = generate_markdown(report)
        assert "Peak RAM" in md
        assert "GPU Mem" in md

    def test_peak_ram_column_absent_when_missing(self) -> None:
        report = BenchmarkReport(
            comparisons=[
                ExperimentComparison(
                    task="detection",
                    model="yolox_s",
                    dataset="pothole",
                    scenario="default",
                    current_metrics={"training:val/mAP": 0.85},
                    baseline_metrics=None,
                ),
            ],
            failures=[],
        )
        md = generate_markdown(report)
        assert "Peak RAM" not in md

    def test_reports_training_iteration_time_not_end_to_end_time(self) -> None:
        report = BenchmarkReport(
            comparisons=[
                ExperimentComparison(
                    task="detection",
                    model="yolox_s",
                    dataset="pothole",
                    scenario="default",
                    current_metrics={
                        "time/train/iter": 0.125,
                        "time/training/e2e": 3600.0,
                    },
                    baseline_metrics=None,
                ),
            ],
            failures=[],
        )

        md = generate_markdown(report)

        assert "Train Iteration Time" in md
        assert "125.0 ms" in md
        assert "Train Time" not in md


# ---------------------------------------------------------------------------
# generate_csv
# ---------------------------------------------------------------------------


class TestGenerateCSV:
    def test_writes_csv_file(self, tmp_path: Path) -> None:
        report = BenchmarkReport(
            comparisons=[
                ExperimentComparison(
                    task="det",
                    model="m",
                    dataset="d",
                    scenario="default",
                    current_metrics={"acc": 0.85},
                    baseline_metrics={"acc": 0.80},
                ),
            ],
            failures=[],
            branch="develop",
            git_sha="abc",
        )
        csv_path = tmp_path / "test.csv"
        generate_csv(report, csv_path)

        assert csv_path.exists()
        content = csv_path.read_text()
        assert "det" in content
        assert "0.85" in content

    def test_empty_report_no_file(self, tmp_path: Path) -> None:
        report = BenchmarkReport(comparisons=[], failures=[])
        csv_path = tmp_path / "empty.csv"
        generate_csv(report, csv_path)
        assert not csv_path.exists()  # no rows to write


# ---------------------------------------------------------------------------
# write_failures_json
# ---------------------------------------------------------------------------


class TestWriteFailuresJson:
    def test_writes_json(self, tmp_path: Path) -> None:
        failures = [
            FailureRecord(task="det", model="m", dataset="d", scenario="default", seed=0, error="OOM"),
        ]
        path = tmp_path / "failures.json"
        write_failures_json(failures, path)

        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["error"] == "OOM"

    def test_includes_failed_phase(self, tmp_path: Path) -> None:
        failures = [
            FailureRecord(
                task="det",
                model="m",
                dataset="d",
                scenario="default",
                seed=0,
                error="OOM",
                failed_phase="optimize",
            ),
        ]
        path = tmp_path / "failures.json"
        write_failures_json(failures, path)

        import json

        data = json.loads(path.read_text())
        assert data[0]["failed_phase"] == "optimize"


class TestGenerateReportIntegration:
    def test_creates_all_output_files(self, tmp_path: Path) -> None:
        results = [
            ExperimentResult(
                task="det",
                model="m",
                dataset="d",
                scenario="default",
                seed=0,
                success=True,
                phases=[PhaseResult(phase="train", metrics={"training:val/mAP": 0.85})],
            ),
        ]
        failures = [
            ExperimentResult.failure(
                task="det",
                model="m2",
                dataset="d",
                scenario="default",
                seed=0,
                exc=RuntimeError("boom"),
            ),
        ]
        baselines: dict[str, dict[str, float] | None] = {"det/m/d/default": None}
        criteria_by_task = {"det": CriteriaConfig(accuracy_metric="mAP", thresholds={})}

        report = generate_report(
            results=results,
            failures=failures,
            baselines=baselines,
            criteria_by_task=criteria_by_task,
            output_dir=tmp_path / "output",
            branch="develop",
            git_sha="abc",
        )

        assert (tmp_path / "output" / "report.md").exists()
        assert (tmp_path / "output" / "aggregated.csv").exists()
        assert (tmp_path / "output" / "failed_experiments.json").exists()
        assert report.branch == "develop"
