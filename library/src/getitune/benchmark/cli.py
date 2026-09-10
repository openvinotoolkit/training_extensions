# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CLI for the benchmark runner: ``python -m getitune.benchmark <subcommand>``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _parse_key_value_pairs(items: list[str] | None) -> dict[str, str]:
    """Parse a list of ``KEY=VALUE`` strings into a dict."""
    if not items:
        return {}
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            msg = f"Invalid KEY=VALUE format: {item!r}"
            raise argparse.ArgumentTypeError(msg)
        key, _, value = item.partition("=")
        result[key] = value
    return result


def _normalize_data_group_filter(data_group: str | None) -> list[str] | None:
    """Convert the user-facing CLI value into backend filter semantics."""
    if data_group in (None, "all"):
        return None
    return [data_group]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by multiple sub-commands."""
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("benchmark_catalog.yaml"),
        help="Path to benchmark_catalog.yaml.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory for dataset storage.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (overrides the GETITUNE_LOG_LEVEL environment variable; default: INFO).",
    )


def _add_tracking_args(parser: argparse.ArgumentParser) -> None:
    """Arguments for MLflow tracking (shared by ``run`` and ``report``)."""
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI (local path or remote server URL).",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="",
        help="Git branch name (auto-detected if empty).",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="manual",
        help="Benchmark trigger type (manual, nightly, weekly, pr).",
    )
    parser.add_argument(
        "--baseline-branch",
        type=str,
        default="develop",
        help="Branch to use for baseline resolution.",
    )


def _build_parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="python -m getitune.benchmark",
        description="GetiTune Automated Model Benchmarking",
    )
    sub = top.add_subparsers(dest="command", required=True)

    # -- provision ---------------------------------------------------------
    prov = sub.add_parser("provision", help="Download & verify datasets.")
    _add_common_args(prov)
    prov.add_argument("--dataset", type=str, nargs="*", default=None, help="Filter by dataset name(s).")
    prov.add_argument("--size-tier", type=str, nargs="*", default=None, help="Filter by size tier(s).")
    prov.add_argument(
        "--data-group",
        type=str,
        default="all",
        choices=("weekly", "extended", "all"),
        help="Filter by data group (weekly, extended, all; default: all).",
    )

    # -- run ---------------------------------------------------------------
    run = sub.add_parser("run", help="Execute benchmark experiments.")
    _add_common_args(run)
    _add_tracking_args(run)
    run.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark_manifest.yaml"),
        help="Path to benchmark_manifest.yaml.",
    )
    run.add_argument("--output-root", type=Path, default=Path("results"), help="Output directory for results.")
    run.add_argument("--task", type=str, nargs="*", default=None, help="Task filter.")
    run.add_argument("--model", type=str, nargs="*", default=None, help="Model name filter.")
    run.add_argument("--dataset", type=str, nargs="*", default=None, help="Dataset name filter.")
    run.add_argument("--priority", type=str, nargs="*", default=None, help="Model priority filter.")
    run.add_argument("--size-tier", type=str, nargs="*", default=None, help="Dataset size tier filter.")
    run.add_argument(
        "--data-group",
        type=str,
        default="all",
        choices=("weekly", "extended", "all"),
        help="Dataset data group filter (weekly, extended, all; default: all).",
    )
    run.add_argument("--scenario", type=str, nargs="*", default=None, help="Scenario name filter.")
    run.add_argument("--scenario-tag", type=str, nargs="*", default=None, help="Scenario tag filter.")
    run.add_argument("--accelerator", type=str, default="gpu", help="Device: gpu, xpu, or cpu.")
    run.add_argument(
        "--openvino-device",
        type=str,
        default=None,
        help="OpenVINO target device, e.g. CPU, GPU, GPU.1, AUTO (default follows accelerator).",
    )
    run.add_argument(
        "--training-device-name",
        type=str,
        default=None,
        help="Physical training device name override when the driver does not expose its marketed name.",
    )
    run.add_argument(
        "--openvino-full-device-name",
        dest="openvino_device_name",
        type=str,
        default=None,
        help=(
            "Human-readable OpenVINO device name for the report; overrides the benchmark_app property name. "
            "Distinct from --openvino-device, which selects the inference target."
        ),
    )
    run.add_argument(
        "--benchmark",
        dest="enable_openvino_benchmark",
        action="store_true",
        help="Run OpenVINO benchmark_app in throughput and latency modes.",
    )
    run.add_argument(
        "--skip-test",
        dest="enable_validation",
        action="store_false",
        default=True,
        help=(
            "Skip the accuracy test phases; the pipeline still runs training, export, INT8 optimization, "
            "and benchmark_app measurements."
        ),
    )
    run.add_argument("--num-seeds", type=int, default=None, help="Override number of seeds.")
    run.add_argument(
        "--max-attempts", type=int, default=3, help="Maximum attempts per experiment (use 1 to disable retries)."
    )
    run.add_argument("--max-epochs", type=int, default=None, help="Override max training epochs.")
    run.add_argument("--eval-upto", type=str, choices=["train", "export", "optimize"], default=None)
    run.add_argument("--deterministic", action="store_const", const=True, default=None)
    run.add_argument("--no-deterministic", dest="deterministic", action="store_const", const=False)
    run.add_argument("--dry-run", action="store_true", default=False, help="Print what would run without executing.")
    run.add_argument(
        "--no-tracking", dest="enable_tracking", action="store_false", default=True, help="Disable MLflow tracking."
    )
    run.add_argument(
        "--no-report",
        dest="enable_report",
        action="store_false",
        default=True,
        help="Disable automatic report generation after the run.",
    )
    run.add_argument(
        "--keep-checkpoints",
        action="store_true",
        default=False,
        help="Keep training checkpoints and exported models after the run (default: delete to save disk).",
    )
    run.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Ad-hoc config overrides passed to the engine (e.g. model.init_args.optimizer.init_args.lr=0.01).",
    )
    run.add_argument(
        "--train-kwarg",
        type=str,
        nargs="*",
        default=None,
        metavar="KEY=VALUE",
        help="Extra keyword arguments forwarded to engine.train() (e.g. precision=32).",
    )
    run.add_argument(
        "--rotation-group",
        type=int,
        default=None,
        help="Rotation group index for extended models. Only models assigned to this group will run.",
    )
    run.add_argument(
        "--no-rotation",
        action="store_true",
        default=False,
        help="Disable rotation logic and run all models regardless of group assignment.",
    )

    # -- report ------------------------------------------------------------
    rpt = sub.add_parser("report", help="Generate a report from existing MLflow data or result files.")
    _add_common_args(rpt)
    _add_tracking_args(rpt)
    rpt.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark_manifest.yaml"),
        help="Path to benchmark_manifest.yaml (for threshold definitions).",
    )
    rpt.add_argument("--output-root", type=Path, default=Path("results"), help="Directory containing results.")
    rpt.add_argument("--output", type=Path, default=None, help="Output report file (default: <output-root>/report.md).")
    rpt.add_argument("--accelerator", type=str, default="gpu", help="Accelerator label for baseline query.")

    # -- clean -------------------------------------------------------------
    cln = sub.add_parser("clean", help="Purge old MLflow runs to free storage (retention policy).")
    cln.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (overrides the GETITUNE_LOG_LEVEL environment variable; default: INFO).",
    )
    cln.add_argument(
        "--mlflow-uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI.",
    )
    cln.add_argument(
        "--max-age-days",
        type=int,
        default=365,
        help="Delete runs older than this many days (default: 365).",
    )
    cln.add_argument(
        "--protect-branch",
        type=str,
        nargs="*",
        default=["develop"],
        help="Branches whose runs are never deleted (default: develop).",
    )
    cln.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what would be deleted without actually deleting.",
    )

    return top


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def _cmd_provision(args: argparse.Namespace) -> int:
    from getitune.benchmark.catalog import load_catalog, provision_datasets

    catalog = load_catalog(args.catalog)
    names = set(args.dataset) if args.dataset else None
    data_groups = _normalize_data_group_filter(args.data_group)
    entries = catalog.filter(size_tiers=args.size_tier, data_groups=data_groups, names=names)
    if not entries:
        logger.warning("No datasets match the given filters.")
        return 0

    logger.info("Provisioning %d dataset(s)…", len(entries))
    provision_datasets(catalog, args.data_root, entries=entries)
    logger.info("Done.")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from getitune.benchmark.catalog import load_catalog
    from getitune.benchmark.manifest import ManifestFilters, load_manifest
    from getitune.benchmark.runner import BenchmarkRunner, RunConfig

    catalog = load_catalog(args.catalog)
    manifest = load_manifest(args.manifest)

    filters = ManifestFilters(
        tasks=args.task,
        models=args.model,
        datasets=args.dataset,
        priorities=args.priority,
        size_tiers=args.size_tier,
        data_groups=_normalize_data_group_filter(args.data_group),
        scenarios=args.scenario,
        scenario_tags=args.scenario_tag,
        dry_run=args.dry_run,
    )

    # Parse ad-hoc overrides and train kwargs
    ad_hoc_overrides = _parse_key_value_pairs(args.override)
    ad_hoc_train_kwargs = _parse_key_value_pairs(args.train_kwarg)

    # CLI flag wins when set; otherwise honour manifest default.
    deterministic = args.deterministic if args.deterministic is not None else manifest.defaults.deterministic

    config = RunConfig(
        manifest_path=args.manifest,
        catalog_path=args.catalog,
        data_root=args.data_root,
        output_root=args.output_root,
        accelerator=args.accelerator,
        deterministic=deterministic,
        max_epochs=args.max_epochs,
        num_seeds=args.num_seeds,
        max_attempts=args.max_attempts,
        eval_upto=args.eval_upto,
        filters=filters,
        mlflow_tracking_uri=args.mlflow_uri,
        branch=args.branch,
        trigger=args.trigger,
        baseline_branch=args.baseline_branch,
        enable_tracking=args.enable_tracking,
        enable_report=args.enable_report,
        keep_checkpoints=args.keep_checkpoints,
        ad_hoc_overrides=ad_hoc_overrides,
        ad_hoc_train_kwargs=ad_hoc_train_kwargs,
        openvino_device=args.openvino_device,
        training_device_name=args.training_device_name,
        openvino_device_name=args.openvino_device_name,
        enable_openvino_benchmark=args.enable_openvino_benchmark,
        enable_validation=args.enable_validation,
        rotation_group=args.rotation_group,
        no_rotation=args.no_rotation,
    )

    runner = BenchmarkRunner(config)
    successes, failures = runner.run(manifest, catalog)

    if failures:
        logger.error("%d experiment(s) failed:", len(failures))
        for f in failures:
            logger.error("  %s/%s/%s seed=%d — %s", f.task, f.model, f.dataset, f.seed, f.error)
        return 1
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Generate a report from existing MLflow data."""
    import mlflow

    from getitune.benchmark.manifest import load_manifest
    from getitune.benchmark.report import generate_report
    from getitune.benchmark.tracking import (
        BenchmarkTracker,
        TrackingConfig,
        get_benchmark_run_id,
        get_git_branch,
        get_git_sha,
    )

    manifest = load_manifest(args.manifest)

    # Set up tracker for baseline resolution
    tracking_config = TrackingConfig(
        tracking_uri=args.mlflow_uri,
        branch=args.branch,
        trigger=args.trigger,
        accelerator=args.accelerator,
        baseline_branch=args.baseline_branch,
    )
    tracker = BenchmarkTracker(tracking_config)
    tracker.setup()

    branch = args.branch or get_git_branch()
    git_sha = get_git_sha()

    # Query MLflow for the successful AND failed seed runs produced by *this*
    # invocation. Skip "rollup" parent runs — only leaf "seed" runs carry
    # per-run status/metrics. Scope to the current benchmark run id so that
    # seed runs left in the persistent tracking store by *previous* executions
    # (which share the same branch/trigger experiment) don't leak into this
    # report — otherwise ``aggregate_metrics_across_seeds`` would average them
    # into the current numbers.
    benchmark_run_id = get_benchmark_run_id()
    client = mlflow.tracking.MlflowClient(args.mlflow_uri)

    experiment_name = tracking_config.experiment_name
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.error("MLflow experiment '%s' not found.", experiment_name)
        return 1

    success_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f"tags.status = 'success' AND tags.run_type = 'seed' AND tags.benchmark_run_id = '{benchmark_run_id}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    failed_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=(
            f"tags.status = 'failed' AND tags.run_type = 'seed' AND tags.benchmark_run_id = '{benchmark_run_id}'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )

    if not success_runs and not failed_runs:
        logger.warning("No runs found in experiment '%s'.", experiment_name)
        return 0

    # Convert MLflow runs into ExperimentResult objects for reuse by report.py
    from getitune.benchmark.experiment import ExperimentResult, PhaseResult

    # Map metric-key prefixes back to the pseudo-phase that emitted them so
    # the regenerated report retains the per-phase grouping (otherwise every
    # metric collapses into a single "all" phase and downstream code that
    # iterates ``result.phases`` loses its structure).
    phase_prefixes: dict[str, str] = {
        "training:": "train",
        "torch:": "test/torch",
        "export:": "export",
        "optimize:": "optimize",
    }

    results: list[ExperimentResult] = []
    for run in success_runs:
        tags = run.data.tags
        metrics = dict(run.data.metrics)
        # MLflow stores total wall time under ``duration_seconds`` (see
        # ``BenchmarkTracker.log_run``). Pop it out so it doesn't leak into a
        # phase metrics dict.
        total_wall = float(metrics.pop("duration_seconds", 0.0))

        bucketed: dict[str, dict[str, float]] = {phase: {} for phase in phase_prefixes.values()}
        leftover: dict[str, float] = {}
        for k, v in metrics.items():
            phase = next((p for prefix, p in phase_prefixes.items() if k.startswith(prefix)), None)
            if phase is None:
                leftover[k] = v
            else:
                bucketed[phase][k] = v

        phases = [PhaseResult(phase=phase, metrics=mtr, wall_time=0.0) for phase, mtr in bucketed.items() if mtr]
        if leftover:
            phases.append(PhaseResult(phase="other", metrics=leftover, wall_time=0.0))
        # Attribute the full run wall time to the train phase (best effort —
        # MLflow no longer carries per-phase timings after rewriting).
        if total_wall and phases:
            phases[0].wall_time = total_wall

        results.append(
            ExperimentResult(
                task=tags.get("task", "unknown"),
                model=tags.get("model", "unknown"),
                dataset=tags.get("dataset", "unknown"),
                scenario=tags.get("scenario", "default"),
                seed=int(tags.get("seed", "0")),
                success=True,
                phases=phases,
            )
        )

    failures: list[ExperimentResult] = []
    for run in failed_runs:
        tags = run.data.tags

        error_summary = tags.get("error", "Unknown error")

        # The full traceback is stored as an artifact (see
        # ``BenchmarkTracker.log_run``) rather than a tag, to keep tag values
        # short. Best-effort download; a missing or unreachable artifact must
        # not break the report. This commonly fails when the MLflow server
        # uses a local-filesystem artifact store (``artifact_uri`` without a
        # scheme) that the reporting host cannot read, in which case the
        # artifact endpoint returns an error. Log the reason at WARNING so the
        # failure is diagnosable, and fall back to the one-line ``error`` tag
        # so the report still surfaces the failure cause instead of silently
        # dropping the entire Tracebacks section.
        traceback_text: str | None = None
        try:
            local_path = client.download_artifacts(run.info.run_id, "traceback.txt")
            traceback_text = Path(local_path).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # never let a bad or unreachable artifact break the report
            logger.warning(
                "Could not download traceback artifact for run %s (%s on %s): %s. Falling back to the short error tag.",
                run.info.run_id,
                tags.get("model", "unknown"),
                tags.get("dataset", "unknown"),
                exc,
            )
            traceback_text = error_summary

        failures.append(
            ExperimentResult(
                task=tags.get("task", "unknown"),
                model=tags.get("model", "unknown"),
                dataset=tags.get("dataset", "unknown"),
                scenario=tags.get("scenario", "default"),
                seed=int(tags.get("seed", "0")),
                success=False,
                phases=[],
                error=error_summary,
                traceback=traceback_text,
                failed_phase=tags.get("error_phase"),
            )
        )

    # Resolve baselines using the tracker
    baselines = tracker.resolve_baselines_for_results(results)

    # Build criteria mapping by task
    criteria_by_task = {task_key: section.criteria for task_key, section in manifest.experiments.items()}

    output_dir = args.output_root
    report = generate_report(
        results=results,
        failures=failures,
        baselines=baselines,
        criteria_by_task=criteria_by_task,
        output_dir=output_dir,
        branch=branch,
        git_sha=git_sha,
        accelerator=args.accelerator,
    )

    # If user specified a custom output path, also copy the markdown there
    if args.output:
        from getitune.benchmark.report import generate_markdown

        md_content = generate_markdown(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md_content)
        logger.info("Wrote report to %s", args.output)

    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    """Purge old MLflow runs according to retention policy."""
    from getitune.benchmark.tracking import BenchmarkTracker, TrackingConfig

    config = TrackingConfig(tracking_uri=args.mlflow_uri)
    tracker = BenchmarkTracker(config)

    deleted = tracker.purge_old_runs(
        max_age_days=args.max_age_days,
        protect_branches=tuple(args.protect_branch),
        dry_run=args.dry_run,
    )

    if args.dry_run:
        logger.info("Dry run complete. %d run(s) would be deleted.", deleted)
    else:
        logger.info("Retention cleanup complete. %d run(s) deleted.", deleted)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for ``python -m getitune.benchmark``."""
    import os

    parser = _build_parser()
    args = parser.parse_args()

    # Resolve log level: explicit --log-level wins, then GETITUNE_LOG_LEVEL,
    # defaulting to INFO. The env var is propagated so spawned subprocess
    # workers honour the same level.
    explicit = getattr(args, "log_level", None)
    level_name: str = explicit if explicit else os.environ.get("GETITUNE_LOG_LEVEL", "INFO")
    os.environ["GETITUNE_LOG_LEVEL"] = level_name

    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "provision": _cmd_provision,
        "run": _cmd_run,
        "report": _cmd_report,
        "clean": _cmd_clean,
    }
    handler = handlers[args.command]
    rc = handler(args)
    sys.exit(rc)
