# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Report generation and regression detection for benchmark runs."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from getitune.benchmark.tracking import rewrite_metric_key

if TYPE_CHECKING:
    from pathlib import Path

    from getitune.benchmark.experiment import ExperimentResult
    from getitune.benchmark.manifest import CriteriaConfig, Threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionResult:
    """Comparison of a single metric against its baseline."""

    metric: str
    current_value: float
    baseline_value: float | None
    margin: float
    direction: str  # "higher_is_better" | "lower_is_better"
    status: str  # "pass" | "regression" | "improvement" | "no_baseline"

    @property
    def delta(self) -> float | None:
        """Relative change from baseline (positive = increased)."""
        if self.baseline_value is None or self.baseline_value == 0:
            return None
        return (self.current_value - self.baseline_value) / abs(self.baseline_value)

    @property
    def delta_pct(self) -> str:
        """Human-readable percentage change."""
        d = self.delta
        if d is None:
            return "N/A"
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1%}"


@dataclass
class ExperimentComparison:
    """Full comparison for one experiment (averaged across seeds)."""

    task: str
    model: str
    dataset: str
    scenario: str
    current_metrics: dict[str, float]
    baseline_metrics: dict[str, float] | None
    regressions: list[RegressionResult] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique identifier: ``task/model/dataset/scenario``."""
        return f"{self.task}/{self.model}/{self.dataset}/{self.scenario}"

    @property
    def has_regression(self) -> bool:
        """Return ``True`` if any threshold check detected a regression."""
        return any(r.status == "regression" for r in self.regressions)

    @property
    def has_baseline(self) -> bool:
        """Return ``True`` if baseline metrics are available."""
        return self.baseline_metrics is not None

    @property
    def overall_status(self) -> str:
        """Aggregate status across all regression checks."""
        if not self.has_baseline:
            return "no_baseline"
        if self.has_regression:
            return "regression"
        return "pass"

    @property
    def status_emoji(self) -> str:
        """Emoji representation of :meth:`overall_status`."""
        return {
            "pass": "✅",
            "regression": "⚠️",
            "no_baseline": "🆕",
        }.get(self.overall_status, "❓")


@dataclass
class FailureRecord:
    """A single failed experiment for the report."""

    task: str
    model: str
    dataset: str
    scenario: str
    seed: int
    error: str
    traceback: str | None = None
    failed_phase: str | None = None  # e.g. "train", "export", "optimize"


@dataclass
class BenchmarkReport:
    """All data needed to render a benchmark report."""

    comparisons: list[ExperimentComparison]
    failures: list[FailureRecord]
    branch: str = "unknown"
    git_sha: str = "unknown"
    accelerator: str = "gpu"
    timestamp: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    @property
    def total_experiments(self) -> int:
        """Total number of experiments (comparisons + failures)."""
        return len(self.comparisons) + len(self.failures)

    @property
    def regression_count(self) -> int:
        """Number of comparisons that have at least one regression."""
        return sum(1 for c in self.comparisons if c.has_regression)

    @property
    def pass_count(self) -> int:
        """Number of comparisons that passed all checks."""
        return sum(1 for c in self.comparisons if c.overall_status == "pass")

    @property
    def no_baseline_count(self) -> int:
        """Number of comparisons with no baseline available."""
        return sum(1 for c in self.comparisons if not c.has_baseline)


# ---------------------------------------------------------------------------
# Regression checking
# ---------------------------------------------------------------------------


def _direction_for_metric(compare: str) -> str:
    """Map a threshold's ``compare`` field to a direction string."""
    return "higher_is_better" if compare == ">=" else "lower_is_better"


def check_regressions(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    thresholds: dict[str, Threshold],
) -> list[RegressionResult]:
    """Compare current metrics against a baseline using manifest thresholds.

    Args:
        current_metrics: Metrics from the current run (averaged across seeds).
        baseline_metrics: Metrics from the baseline run, or ``None``.
        thresholds: Dict of ``{metric_key: Threshold}`` from the manifest.

    Returns:
        A :class:`RegressionResult` for each threshold that can be evaluated.
    """
    results: list[RegressionResult] = []

    for metric_key, threshold in thresholds.items():
        compare = threshold.compare
        margin = threshold.margin
        direction = _direction_for_metric(compare)

        # Normalize threshold key into the same shape used by current/baseline dicts.
        lookup_key = rewrite_metric_key(metric_key)

        current_value = current_metrics.get(lookup_key)
        if current_value is None:
            # Metric not present in current run — skip silently
            continue

        if baseline_metrics is None:
            results.append(
                RegressionResult(
                    metric=metric_key,
                    current_value=current_value,
                    baseline_value=None,
                    margin=margin,
                    direction=direction,
                    status="no_baseline",
                )
            )
            continue

        baseline_value = baseline_metrics.get(lookup_key)
        if baseline_value is None:
            results.append(
                RegressionResult(
                    metric=metric_key,
                    current_value=current_value,
                    baseline_value=None,
                    margin=margin,
                    direction=direction,
                    status="no_baseline",
                )
            )
            continue

        # Determine status
        if direction == "higher_is_better":
            # Regression if current < baseline * (1 - margin)
            threshold_value = baseline_value * (1 - margin)
            if current_value < threshold_value:
                status = "regression"
            elif current_value > baseline_value * (1 + margin):
                status = "improvement"
            else:
                status = "pass"
        else:
            # lower_is_better: regression if current > baseline * (1 + margin)
            threshold_value = baseline_value * (1 + margin)
            if current_value > threshold_value:
                status = "regression"
            elif current_value < baseline_value * (1 - margin):
                status = "improvement"
            else:
                status = "pass"

        results.append(
            RegressionResult(
                metric=metric_key,
                current_value=current_value,
                baseline_value=baseline_value,
                margin=margin,
                direction=direction,
                status=status,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Metric aggregation (average across seeds)
# ---------------------------------------------------------------------------


def aggregate_metrics_across_seeds(
    results: list[ExperimentResult],
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    """Average metrics across seeds for each ``(task, model, dataset, scenario)`` group.

    Per-seed metric keys are normalized via :func:`rewrite_metric_key` so they
    share the same shape as MLflow-stored baseline metrics.

    Returns:
        ``{(task, model, dataset, scenario): averaged_metrics}``
    """
    grouped: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)

    for result in results:
        if not result.success:
            continue
        key = (result.task, result.model, result.dataset, result.scenario)
        normalized = {rewrite_metric_key(k): v for k, v in result.all_metrics().items()}
        grouped[key].append(normalized)

    averaged: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for key, metric_dicts in grouped.items():
        if not metric_dicts:
            continue
        all_keys = set()
        for d in metric_dicts:
            all_keys.update(d.keys())

        avg: dict[str, float] = {}
        for mk in all_keys:
            values = [d[mk] for d in metric_dicts if mk in d]
            if values:
                avg[mk] = sum(values) / len(values)
        averaged[key] = avg

    return averaged


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(
    results: list[ExperimentResult],
    failures: list[ExperimentResult],
    baselines: dict[str, dict[str, float] | None],
    criteria_by_task: dict[str, CriteriaConfig],
    *,
    branch: str = "unknown",
    git_sha: str = "unknown",
    accelerator: str = "gpu",
) -> BenchmarkReport:
    """Build a :class:`BenchmarkReport` from run results and baselines.

    Args:
        results: Successful :class:`ExperimentResult` instances.
        failures: Failed :class:`ExperimentResult` instances.
        baselines: ``{key: baseline_metrics}`` from :meth:`BenchmarkTracker.resolve_baselines_for_results`.
        criteria_by_task: ``{task_key: CriteriaConfig}`` from the manifest.
        branch: Git branch.
        git_sha: Git commit SHA.
        accelerator: Accelerator label.
    """
    averaged = aggregate_metrics_across_seeds(results)

    comparisons: list[ExperimentComparison] = []
    for (task, model, dataset, scenario), current_metrics in averaged.items():
        baseline_key = f"{task}/{model}/{dataset}/{scenario}"
        baseline_metrics = baselines.get(baseline_key)
        criteria = criteria_by_task.get(task)
        thresholds = criteria.thresholds if criteria else {}

        regressions = check_regressions(current_metrics, baseline_metrics, thresholds)

        comparisons.append(
            ExperimentComparison(
                task=task,
                model=model,
                dataset=dataset,
                scenario=scenario,
                current_metrics=current_metrics,
                baseline_metrics=baseline_metrics,
                regressions=regressions,
            )
        )

    failure_records = [
        FailureRecord(
            task=f.task,
            model=f.model,
            dataset=f.dataset,
            scenario=f.scenario,
            seed=f.seed,
            error=f.error or "Unknown error",
            traceback=getattr(f, "traceback", None),
            failed_phase=getattr(f, "failed_phase", None),
        )
        for f in failures
    ]

    return BenchmarkReport(
        comparisons=comparisons,
        failures=failure_records,
        branch=branch,
        git_sha=git_sha,
        accelerator=accelerator,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_metric(value: float | None, precision: int = 4) -> str:
    """Format a metric value for the Markdown table."""
    if value is None:
        return "—"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.{precision}f}"


def _fmt_time(seconds: float | None) -> str:
    """Format seconds into a human-readable duration."""
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def _fmt_memory(mb: float | None) -> str:
    """Format memory in MB into a human-readable string."""
    if mb is None:
        return "—"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _fmt_latency(seconds: float | None) -> str:
    """Format a per-sample latency into a human-readable string.

    Unlike :func:`_fmt_time` (built for run-length durations such as train
    time), per-sample test latency is typically well under a second. Reusing
    ``_fmt_time``'s single-decimal-second precision collapses nearly every
    real value into ``0.0s`` or ``0.1s``, so sub-second values are rendered
    in milliseconds instead to preserve meaningful precision.
    """
    if seconds is None:
        return "—"
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f}s"


def generate_markdown(report: BenchmarkReport) -> str:
    """Render a :class:`BenchmarkReport` as a Markdown string."""
    lines: list[str] = []

    # Header
    lines.append(f"## GetiTune Benchmark Report — {report.timestamp}")
    lines.append("")
    lines.append(
        f"**Branch:** `{report.branch}` | **Commit:** `{report.git_sha}` | **Accelerator:** {report.accelerator}"
    )
    lines.append("")

    # Summary stats
    lines.append(
        f"**{report.pass_count}** passed · "
        f"**{report.regression_count}** regressions · "
        f"**{report.no_baseline_count}** new (no baseline) · "
        f"**{len(report.failures)}** failed"
    )
    lines.append("")

    # Group comparisons by task
    by_task: dict[str, list[ExperimentComparison]] = defaultdict(list)
    for comp in report.comparisons:
        by_task[comp.task].append(comp)

    for task, comps in sorted(by_task.items()):
        lines.append(f"### {task}")
        lines.append("")

        # Build table header — find which metric columns are available
        # Use a standard set: primary accuracy metric, train iteration time, GPU mem, test latency
        metric_cols = _detect_metric_columns(comps)

        header = "| Model | Dataset | Status "
        separator = "| --- | --- | --- "
        for col_label, _, _ in metric_cols:
            header += f"| {col_label} "
            separator += "| --- "
        header += "|"
        separator += "|"
        lines.append(header)
        lines.append(separator)

        for comp in sorted(comps, key=lambda c: (c.model, c.dataset, c.scenario)):
            # Status column with regression details
            status_detail = comp.status_emoji
            regression_details = [r for r in comp.regressions if r.status == "regression"]
            if regression_details:
                detail_parts = []
                for r in regression_details:
                    short_metric = r.metric.rsplit("/", 1)[-1] if "/" in r.metric else r.metric
                    detail_parts.append(f"{short_metric} {r.delta_pct}")
                status_detail += " " + ", ".join(detail_parts)

            row = f"| {comp.model} | {comp.dataset} | {status_detail} "

            for _col_label, metric_key, fmt_fn in metric_cols:
                value = comp.current_metrics.get(metric_key)
                formatted = fmt_fn(value)

                # Add delta annotation if baseline exists
                delta_str = ""
                if comp.baseline_metrics and value is not None:
                    bv = comp.baseline_metrics.get(metric_key)
                    if bv is not None and bv != 0:
                        delta = (value - bv) / abs(bv)
                        sign = "+" if delta >= 0 else ""
                        delta_str = f" ({sign}{delta:.1%})"

                row += f"| {formatted}{delta_str} "

            row += "|"
            lines.append(row)

        lines.append("")

    # Regression alerts section
    regressions = [c for c in report.comparisons if c.has_regression]
    if regressions:
        lines.append("### ⚠️ Regression Alerts")
        lines.append("")
        for comp in regressions:
            lines.append(f"<details><summary><b>{comp.model}</b> on <b>{comp.dataset}</b> ({comp.scenario})</summary>")
            lines.append("")
            lines.append("| Metric | Current | Baseline | Δ | Margin | Status |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            lines.extend(
                f"| `{r.metric}` "
                f"| {_fmt_metric(r.current_value)} "
                f"| {_fmt_metric(r.baseline_value)} "
                f"| {r.delta_pct} "
                f"| ±{r.margin:.0%} "
                f"| ❌ regression |"
                for r in comp.regressions
                if r.status == "regression"
            )
            lines.append("")
            lines.append("</details>")
            lines.append("")

    # Failures section - deduplicate by (model, dataset, scenario, error) so
    # that multiple seeds producing the same error are shown as a single row.
    if report.failures:
        seen: set[tuple[str, str, str, str]] = set()
        unique: list[tuple[str, str, str, str, str, str | None]] = []
        for f in report.failures:
            error_short = f.error[:120].replace("|", "\\|").replace("\n", " ")
            stage = f.failed_phase or "unknown"
            key = (f.model, f.dataset, f.scenario, error_short)
            if key in seen:
                continue
            seen.add(key)
            unique.append((*key, stage, f.traceback))

        lines.append(f"### ❌ Failures ({len(unique)})")
        lines.append("")
        lines.append("| Model | Dataset | Scenario | Stage | Error |")
        lines.append("| --- | --- | --- | --- | --- |")
        for model, dataset, scenario, error_short, stage, _tb in unique:
            lines.append(f"| {model} | {dataset} | {scenario} | {stage} | {error_short} |")
        lines.append("")

        # Expandable tracebacks below the table
        any_tb = any(tb for *_, tb in unique)
        if any_tb:
            lines.append("#### Tracebacks")
            lines.append("")
            for model, dataset, scenario, _error_short, stage, tb in unique:
                if not tb:
                    continue
                lines.append(
                    f"<details><summary><b>{model}</b> on <b>{dataset}</b> "
                    f"({scenario}) — failed at <b>{stage}</b></summary>"
                )
                lines.append("")
                lines.append("```")
                # Guard against accidental fence-breaking content
                lines.append(tb.rstrip().replace("```", "``\u200b`"))
                lines.append("```")
                lines.append("")
                lines.append("</details>")
                lines.append("")

    return "\n".join(lines)


def _detect_metric_columns(
    comps: list[ExperimentComparison],
) -> list[tuple[str, str, Any]]:
    """Detect which metric columns are present across all comparisons.

    Returns a list of ``(label, metric_key, format_fn)`` tuples.
    """
    # Collect all metric keys across comparisons
    all_keys: set[str] = set()
    for comp in comps:
        all_keys.update(comp.current_metrics.keys())

    columns: list[tuple[str, str, Any]] = []

    # Substrings that identify metrics where lower values are better. The
    # arrow shown next to the column header reflects this so that
    # e.g. ``loss`` and ``iter_time`` columns aren't tagged with ↑.
    lower_is_better_markers = ("loss", "iter_time", "/time", "latency", "e2e_time")

    def _arrow_for(key: str) -> str:
        return "↓" if any(m in key for m in lower_is_better_markers) else "↑"

    # Accuracy-like metrics (val/*)
    for key in sorted(all_keys):
        if "val/" in key and "training:" in key:
            label_short = key.split("val/", 1)[1] if "val/" in key else key
            columns.append((f"Val {label_short} {_arrow_for(key)}", key, _fmt_metric))

    # Test metrics (torch)
    for key in sorted(all_keys):
        if key.startswith("torch:") and "test/" in key and "latency" not in key and "e2e_time" not in key:
            label_short = key.split("test/", 1)[1] if "test/" in key else key
            columns.append((f"Test {label_short} {_arrow_for(key)}", key, _fmt_metric))

    # Export test metrics
    for key in sorted(all_keys):
        if key.startswith("export:") and "test/" in key and "latency" not in key and "e2e_time" not in key:
            label_short = key.split("test/", 1)[1] if "test/" in key else key
            columns.append((f"Export {label_short} {_arrow_for(key)}", key, _fmt_metric))

    # Optimize (quantized) test metrics
    for key in sorted(all_keys):
        if key.startswith("optimize:") and "test/" in key and "latency" not in key and "e2e_time" not in key:
            label_short = key.split("test/", 1)[1] if "test/" in key else key
            columns.append((f"Optimize {label_short} {_arrow_for(key)}", key, _fmt_metric))

    # Training performance is reported as post-warmup per-iteration time, not
    # end-to-end training duration. The latter includes validation,
    # checkpointing, and other work unrelated to a training iteration.
    train_iter_key = rewrite_metric_key("training:train/iter_time")
    if train_iter_key in all_keys:
        columns.append(("Train Iteration Time ↓", train_iter_key, _fmt_latency))

    if "training:gpu_mem" in all_keys:
        columns.append(("GPU Mem ↓", "training:gpu_mem", _fmt_memory))

    if "training:ram_mem" in all_keys:
        columns.append(("Peak RAM ↓", "training:ram_mem", _fmt_memory))

    if rewrite_metric_key("torch:test/latency") in all_keys:
        columns.append(("Test Latency ↓", rewrite_metric_key("torch:test/latency"), _fmt_latency))

    for phase, label in (("export", "FP16"), ("optimize", "INT8")):
        fps_key = rewrite_metric_key(f"{phase}:throughput:fps")
        latency_key = rewrite_metric_key(f"{phase}:latency:latency_ms")
        if fps_key in all_keys:
            columns.append((f"{label} FPS ↑", fps_key, _fmt_metric))
        if latency_key in all_keys:
            columns.append((f"{label} Latency ↓", latency_key, lambda v: _fmt_metric(v, 2)))

    return columns


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


# Per-class metrics whose underlying value is a list; the executor flattens
# them to a scalar ``-1`` sentinel when no per-class data is available, which
# pollutes downstream CSVs (Excel quote-escapes them as ``'-1``). Mirror the
# filter used in ``tracking.py`` so the report-level CSVs are also clean.
_PER_CLASS_SENTINEL_KEYS: frozenset[str] = frozenset(
    {
        "training:val/map_per_class",
        "training:val/mar_100_per_class",
    }
)


def _drop_constant_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove columns whose value is identical (or always empty) across rows.

    Greatly reduces noise when every run shares the same branch / accelerator
    / dataset / scenario etc., as is the case for a single benchmark
    invocation.
    """
    if len(rows) <= 1:
        return rows
    keys = {k for row in rows for k in row}
    constant: set[str] = set()
    for k in keys:
        seen: set[Any] = set()
        for row in rows:
            v = row.get(k, "")
            seen.add(v)
            if len(seen) > 1:
                break
        if len(seen) <= 1:
            constant.add(k)
    if not constant:
        return rows
    return [{k: v for k, v in row.items() if k not in constant} for row in rows]


def _is_clean_metric(key: str, value: object) -> bool:
    """Return True if ``(key, value)`` should appear in the cleaned CSV."""
    if not isinstance(value, (int, float)):
        return False
    return not (key in _PER_CLASS_SENTINEL_KEYS and value == -1)


def generate_csv(report: BenchmarkReport, output_path: Path) -> None:
    """Write an aggregated CSV file with all results and comparisons.

    The output is optimized for downstream analysis:

    - Only numeric metric values are written (no ``'-1`` per-class sentinels).
    - Columns that have a single constant value across every row are dropped.
    - Failures are emitted with structured ``error_type`` / ``error_phase``
      columns alongside a one-line ``error`` summary; full tracebacks are
      written to ``failed_experiments.json`` only.
    """
    import csv

    rows: list[dict[str, Any]] = []
    for comp in report.comparisons:
        row: dict[str, Any] = {
            "run_type": "aggregate",
            "task": comp.task,
            "model": comp.model,
            "dataset": comp.dataset,
            "scenario": comp.scenario,
            "status": comp.overall_status,
            "branch": report.branch,
            "git_sha": report.git_sha,
            "accelerator": report.accelerator,
        }
        for k, v in sorted(comp.current_metrics.items()):
            if _is_clean_metric(k, v):
                row[f"current:{k}"] = v
        if comp.baseline_metrics:
            for k, v in sorted(comp.baseline_metrics.items()):
                if _is_clean_metric(k, v):
                    row[f"baseline:{k}"] = v
        rows.append(row)

    for f in report.failures:
        error_one_line = (f.error or "").splitlines()[0:1]
        error_short = error_one_line[0].strip() if error_one_line else ""
        error_type = error_short.split(":", 1)[0].strip() if ":" in error_short else "Unknown"
        rows.append(
            {
                "run_type": "failure",
                "task": f.task,
                "model": f.model,
                "dataset": f.dataset,
                "scenario": f.scenario,
                "seed": f.seed,
                "status": "failed",
                "error_type": error_type,
                "error_phase": f.failed_phase or "unknown",
                "error": error_short[:250],
                "branch": report.branch,
                "git_sha": report.git_sha,
                "accelerator": report.accelerator,
            }
        )

    if not rows:
        logger.warning("No results to write to CSV.")
        return

    rows = _drop_constant_columns(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_fields = list(dict.fromkeys(k for row in rows for k in row))
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote CSV report to %s", output_path)


# ---------------------------------------------------------------------------
# Failure log
# ---------------------------------------------------------------------------


def write_failures_json(failures: list[FailureRecord], output_path: Path) -> None:
    """Write structured failure log to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "task": f.task,
            "model": f.model,
            "dataset": f.dataset,
            "scenario": f.scenario,
            "seed": f.seed,
            "failed_phase": f.failed_phase,
            "error": f.error,
            "traceback": f.traceback,
        }
        for f in failures
    ]
    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Wrote %d failure(s) to %s", len(data), output_path)


# ---------------------------------------------------------------------------
# High-level report generation
# ---------------------------------------------------------------------------


def generate_report(
    results: list[ExperimentResult],
    failures: list[ExperimentResult],
    baselines: dict[str, dict[str, float] | None],
    criteria_by_task: dict[str, CriteriaConfig],
    output_dir: Path,
    *,
    branch: str = "unknown",
    git_sha: str = "unknown",
    accelerator: str = "gpu",
) -> BenchmarkReport:
    """Build a report, write Markdown + CSV + failures JSON, and return it.

    This is the main entry point for report generation, typically called
    from the runner or the ``report`` CLI subcommand.
    """
    report = build_report(
        results=results,
        failures=failures,
        baselines=baselines,
        criteria_by_task=criteria_by_task,
        branch=branch,
        git_sha=git_sha,
        accelerator=accelerator,
    )

    # Write Markdown report
    md_content = generate_markdown(report)
    md_path = output_dir / "report.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content)
    logger.info("Wrote Markdown report to %s", md_path)

    # Write CSV
    csv_path = output_dir / "aggregated.csv"
    generate_csv(report, csv_path)

    # Write failures JSON
    if report.failures:
        write_failures_json(report.failures, output_dir / "failed_experiments.json")

    return report
