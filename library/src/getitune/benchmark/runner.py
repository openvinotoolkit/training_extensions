# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark runner — core orchestration loop."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle  # nosec B403 - used only for IPC between parent and child benchmark processes we spawn
import traceback as _traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from getitune.benchmark.catalog import DatasetCatalog, provision_datasets
from getitune.benchmark.experiment import (
    ExperimentExecutor,
    ExperimentResult,
    PhaseExecutionError,
    PhaseResult,
    detect_resume_point,
)
from getitune.benchmark.manifest import (
    BenchmarkManifest,
    Experiment,
    ManifestFilters,
    count_experiments,
    iter_experiments,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# Ordered list of phases that the runner steps through.  Each phase maps
# to a method name on :class:`ExperimentExecutor`.
_PHASE_CHAIN: list[tuple[str, str]] = [
    ("train", "train"),
    ("test/torch", "test_torch"),
    ("export", "export"),
    ("test/export", "test_export"),
    ("benchmark/export", "benchmark_export"),
    ("optimize", "optimize"),
    ("test/optimize", "test_optimize"),
    ("benchmark/optimize", "benchmark_optimize"),
]

# Which phases are included for a given ``eval_upto`` value.
_EVAL_UPTO_GATES: dict[str, set[str]] = {
    "train": {"train", "test/torch"},
    "export": {
        "train",
        "test/torch",
        "export",
        "test/export",
    },
    "optimize": {
        "train",
        "test/torch",
        "export",
        "test/export",
        "optimize",
        "test/optimize",
    },
}

_BENCHMARK_PHASES: dict[str, set[str]] = {
    "export": {"benchmark/export"},
    "optimize": {
        "benchmark/export",
        "benchmark/optimize",
    },
}

_VALIDATION_PHASES = {"test/torch", "test/export", "test/optimize"}
_BENCHMARK_PHASE_NAMES = {"benchmark/export", "benchmark/optimize"}

_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Subprocess isolation
# ---------------------------------------------------------------------------


def _subprocess_worker(
    payload: bytes,
    result_path: str,
) -> None:
    """Entry point for a spawned child that runs a single ``_execute`` call.

    This function is deliberately module-level so it pickles under the
    ``spawn`` start method. It reconstructs the runner in the child, invokes
    ``_execute`` exactly once, and writes the resulting :class:`ExperimentResult`
    (or a failure surrogate) to *result_path* as a pickle file. Any exception
    is captured and also serialized there — the parent never raises from this
    worker, it only inspects the file.
    """
    # Configure logging in the child to mirror the parent's format so output
    # stays readable in the combined stream.
    logging.basicConfig(
        level=os.environ.get("GETITUNE_LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    try:
        config, experiment, seed, data_path, allowed_phases, deterministic = pickle.loads(payload)  # noqa: S301  # nosec B301 - payload is built by the parent process in this same module
        # Build a throw-away runner just to reuse ``_execute``. ``_tracker``
        # stays ``None`` in the child so MLflow writes happen only in the
        # parent after the child returns.
        runner = BenchmarkRunner(config)
        result = runner._execute(  # noqa: SLF001
            experiment,
            seed,
            data_path,
            allowed_phases,
            deterministic=deterministic,
        )
        outcome: tuple[str, object] = ("ok", result)
    except BaseException as exc:
        # Preserve the failing phase (if any) so the parent can reconstruct a
        # PhaseExecutionError and keep stage information across the boundary.
        phase = getattr(exc, "phase", None)
        outcome = ("error", (type(exc).__name__, str(exc), _traceback.format_exc(), phase))

    try:
        with open(result_path, "wb") as fh:  # noqa: PTH123
            pickle.dump(outcome, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # Last-ditch: write a minimal text marker so the parent at least
        # knows the child did start before it died.
        try:
            with open(result_path + ".err", "w") as fh:  # noqa: PTH123
                fh.write("failed to serialize worker outcome\n")
        except Exception:  # noqa: S110
            pass


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------


def _rotation_bucket(name: str, groups: int) -> int:
    """Stable hash of *name* into ``[0, groups)``."""
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % groups


def _cleanup_resources(*, reset_cuda_peak: bool = False) -> None:
    """Best-effort cleanup of leaked resources between phases/experiments.

    The benchmark runs every experiment inside a single long-lived Python
    process. Several downstream libraries leak workers / semaphores / GPU
    memory across runs which, over ~30+ experiments, accumulates until the
    Linux OOM-killer terminates the process without a traceback.

    This helper tries to reclaim:

    * joblib/loky reusable executor workers (NNCF/PTQ spawns 64 of them per
      ``optimize`` phase and does not shut them down).
    * Python garbage (cyclic references to DataLoader workers, Lightning
      trainer, CUDA tensors, …).
    * The CUDA caching allocator's reserved memory.

    The function never raises — any failure is logged at debug level.
    """
    # 1. Shut down the loky reusable executor that NNCF/joblib leaves behind.
    try:
        from joblib.externals.loky import get_reusable_executor

        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        logger.debug("loky executor shutdown failed (ignored).", exc_info=True)

    # 2. Python GC — run twice to break cycles involving __del__.
    gc.collect()
    gc.collect()

    # 3. CUDA cache + optional peak-memory reset.
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            if reset_cuda_peak:
                torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
            if reset_cuda_peak:
                torch.xpu.reset_peak_memory_stats()
    except Exception:
        logger.debug("Accelerator cache cleanup failed (ignored).", exc_info=True)


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """All settings needed to drive a benchmark run."""

    manifest_path: Path
    catalog_path: Path
    data_root: Path
    output_root: Path
    accelerator: str = "gpu"
    deterministic: bool | str = True  # True, False, or "warn" (PyTorch semantics)
    max_epochs: int | None = None
    num_seeds: int | None = None  # override manifest default
    eval_upto: str | None = None  # override manifest default
    filters: ManifestFilters = field(default_factory=ManifestFilters)

    # Tracking & reporting
    mlflow_tracking_uri: str = "./mlruns"
    branch: str = ""
    trigger: str = "manual"
    baseline_branch: str = "develop"
    enable_tracking: bool = True
    enable_report: bool = True

    # Disk management
    keep_checkpoints: bool = False  # If False, delete .ckpt/.xml/.bin after metrics are extracted

    # Ad-hoc overrides (from CLI --override / --train-kwarg flags)
    ad_hoc_overrides: dict[str, str] = field(default_factory=dict)
    ad_hoc_train_kwargs: dict[str, str] = field(default_factory=dict)
    benchmark_app: str | None = None
    openvino_device: str | None = None
    training_device_name: str | None = None
    openvino_device_name: str | None = None
    enable_openvino_benchmark: bool = False
    enable_validation: bool = True

    # Rotation logic
    rotation_group: int | None = None  # If set, only run extended models in this group
    no_rotation: bool = False  # If True, skip rotation filtering entirely

    # Process isolation. When True (default), each seed of each experiment
    # runs in a freshly spawned child Python process that exits as soon as
    # the seed is done.
    isolate_in_subprocess: bool = True
    subprocess_timeout: float | None = None  # seconds; None = no timeout
    max_attempts: int = _MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates the full benchmark run.

    The runner iterates over experiments sequentially (one at a time) so that
    each experiment has exclusive GPU access, producing stable and
    reproducible metrics.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.results: list[ExperimentResult] = []
        self.failures: list[ExperimentResult] = []
        self._tracker = None  # type: ignore[assignment]

    def run(
        self,
        manifest: BenchmarkManifest,
        catalog: DatasetCatalog,
    ) -> tuple[list[ExperimentResult], list[ExperimentResult]]:
        """Execute all matching experiments.

        Returns ``(successes, failures)``.
        """
        # Sync the task → primary-metric lookup with the live manifest.
        from getitune.benchmark.tracking import register_primary_metrics

        register_primary_metrics(
            {task: f"training:val/{section.criteria.accuracy_metric}" for task, section in manifest.experiments.items()}
        )

        # Set up MLflow tracking if enabled
        if self.config.enable_tracking:
            self._setup_tracking()

        eval_upto = self.config.eval_upto or manifest.defaults.eval_upto
        allowed_phases = set(_EVAL_UPTO_GATES.get(eval_upto, _EVAL_UPTO_GATES["optimize"]))
        if self.config.enable_openvino_benchmark:
            allowed_phases.update(_BENCHMARK_PHASES.get(eval_upto, set()))
        if not self.config.enable_validation:
            allowed_phases.difference_update(_VALIDATION_PHASES)

        catalog_names = {e.name for e in catalog.all_entries()}

        # Build a lookup for dataset size tiers (for filtering and MLflow tags)
        size_tier_map: dict[str, str] = {}
        data_group_map: dict[str, str] = {}
        for entry in catalog.all_entries():
            size_tier_map[entry.name] = entry.size_tier
            data_group_map[entry.name] = entry.data_group

        experiments = list(
            iter_experiments(
                manifest,
                self.config.filters,
                catalog_names,
                size_tier_map=size_tier_map,
                data_group_map=data_group_map,
            )
        )

        # Apply rotation logic for extended models
        rotation_groups = manifest.defaults.rotation.get("extended_groups", 0)
        if self.config.rotation_group is not None and not self.config.no_rotation and rotation_groups > 0:
            group = self.config.rotation_group
            experiments = [
                exp
                for exp in experiments
                if exp.model.priority != "extended" or _rotation_bucket(exp.model.name, rotation_groups) == group
            ]
            logger.info(
                "Rotation group %d/%d: %d experiment(s) after filtering.",
                group,
                rotation_groups,
                len(experiments),
            )

        if not experiments:
            logger.warning("No experiments match the current filters.")
            return [], []

        if self.config.filters.dry_run:
            total_unfiltered = count_experiments(manifest)
            logger.info(
                "Dry run: would execute %d experiments (out of %d declared in manifest).",
                len(experiments),
                total_unfiltered,
            )
            for exp in experiments:
                seeds = self.config.num_seeds or exp.num_seeds
                logger.info(
                    "  %s / %s / %s / %s  [priority=%s, eval_upto=%s]  (%d seeds)",
                    exp.task,
                    exp.model.name,
                    exp.dataset_name,
                    exp.scenario.name,
                    exp.model.priority,
                    exp.eval_upto,
                    seeds,
                )
            return [], []

        # Provision required datasets
        required_names = {exp.dataset_name for exp in experiments}
        required_entries = catalog.filter(names=required_names)
        logger.info("Provisioning %d dataset(s)…", len(required_entries))
        dataset_paths = provision_datasets(catalog, self.config.data_root, entries=required_entries)

        total = len(experiments)
        for idx, experiment in enumerate(experiments, 1):
            logger.info(
                "[%d/%d] %s / %s / %s (scenario=%s)",
                idx,
                total,
                experiment.task,
                experiment.model.name,
                experiment.dataset_name,
                experiment.scenario.name,
            )
            data_path = dataset_paths.get(experiment.dataset_name)
            if data_path is None:
                logger.error("Dataset '%s' not provisioned; skipping.", experiment.dataset_name)
                continue

            self._run_experiment(experiment, data_path, allowed_phases, size_tier_map)

            # Reclaim resources before the next experiment runs.
            _cleanup_resources(reset_cuda_peak=True)

            # Persist report after each experiment so partial results survive crashes
            if self.config.enable_report:
                self._generate_report(manifest)

        logger.info(
            "Benchmark complete. %d succeeded, %d failed.",
            len(self.results),
            len(self.failures),
        )

        # Cleanup heavy artifacts to save disk space
        if not self.config.keep_checkpoints:
            self._cleanup_checkpoints()

        return self.results, self.failures

    # -- tracking setup ----------------------------------------------------

    def _setup_tracking(self) -> None:
        """Initialize the MLflow tracker."""
        from getitune.benchmark.tracking import BenchmarkTracker, TrackingConfig

        tracking_config = TrackingConfig(
            tracking_uri=self.config.mlflow_tracking_uri,
            branch=self.config.branch,
            trigger=self.config.trigger,
            accelerator=self.config.accelerator,
            baseline_branch=self.config.baseline_branch,
        )
        self._tracker = BenchmarkTracker(tracking_config)
        self._tracker.setup()
        logger.info("MLflow tracking enabled (uri=%s).", self.config.mlflow_tracking_uri)

    # -- report generation -------------------------------------------------

    def _generate_report(self, manifest: BenchmarkManifest) -> None:
        """Generate a Markdown + CSV report after the run completes."""
        try:
            from getitune.benchmark.report import generate_report
            from getitune.benchmark.tracking import get_git_branch, get_git_sha

            # Resolve baselines from MLflow
            baselines: dict[str, dict[str, float] | None] = {}
            if self._tracker is not None:
                try:
                    baselines = self._tracker.resolve_baselines_for_results(self.results)
                except Exception:
                    logger.warning("Failed to resolve baselines from MLflow.", exc_info=True)

            # Build criteria mapping by task
            criteria_by_task = {task_key: section.criteria for task_key, section in manifest.experiments.items()}

            branch = self.config.branch or get_git_branch()
            git_sha = get_git_sha()

            generate_report(
                results=self.results,
                failures=self.failures,
                baselines=baselines,
                criteria_by_task=criteria_by_task,
                output_dir=self.config.output_root,
                branch=branch,
                git_sha=git_sha,
                accelerator=self.config.accelerator,
            )
        except Exception:
            logger.warning("Failed to generate report.", exc_info=True)

    # -- disk cleanup ------------------------------------------------------

    def _cleanup_checkpoints(self) -> None:
        """Delete training checkpoints and exported models to save disk space.

        Keeps only ``metrics.csv`` and ``result.json`` files which contain
        the extracted metrics.  Skipped when ``RunConfig.keep_checkpoints``
        is ``True``.
        """
        output_root = self.config.output_root
        if not output_root.exists():
            return

        extensions = {".ckpt", ".xml", ".bin", ".onnx"}
        removed = 0
        freed_bytes = 0
        for path in output_root.rglob("*"):
            if self.config.enable_openvino_benchmark and not self._performance_result_complete(path):
                continue
            if path.is_file() and path.suffix in extensions:
                freed_bytes += path.stat().st_size
                path.unlink()
                removed += 1

        if removed:
            freed_mb = freed_bytes / (1024 * 1024)
            logger.info("Cleanup: removed %d checkpoint/model files (%.1f MB freed).", removed, freed_mb)

    @staticmethod
    def _performance_result_complete(path: Path) -> bool:
        """Return whether *path* belongs to a complete performance result."""
        seed_dir = path
        while seed_dir != seed_dir.parent and not (seed_dir / "performance_result.json").exists():
            seed_dir = seed_dir.parent
        result_path = seed_dir / "performance_result.json"
        if not result_path.exists():
            return False
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return isinstance(result, dict) and {"fp16", "int8"} <= result.keys()

    # -- single experiment -------------------------------------------------

    def _run_experiment(
        self,
        experiment: Experiment,
        data_path: Path,
        allowed_phases: set[str],
        size_tier_map: dict[str, str],
    ) -> None:
        """Run all seeds for a single experiment, then log them to MLflow.

        Seeds are executed first and their results collected in memory. Only
        after every seed has completed do we open any MLflow runs and push
        results. This guarantees that partially-completed experiments never
        leak into MLflow: either every seed is logged together (with its
        rollup aggregate), or nothing is.
        """
        num_seeds = self.config.num_seeds or experiment.num_seeds
        seed_results: list[ExperimentResult] = []

        # Phase 1 — run every seed. No MLflow calls happen here.
        for seed in range(num_seeds):
            result = self._run_single(experiment, seed, data_path, allowed_phases)
            seed_results.append(result)
            if result.success:
                self.results.append(result)
            else:
                self.failures.append(result)

        # Phase 2 — all seeds are done; now push everything to MLflow.
        self._log_experiment_results(seed_results, experiment, size_tier_map)

    def _log_experiment_results(
        self,
        seed_results: list[ExperimentResult],
        experiment: Experiment,
        size_tier_map: dict[str, str],
    ) -> None:
        """Push a completed experiment's seed results (and rollup) to MLflow.

        No-op when tracking is disabled or no seeds were produced.

        When more than one seed was run, all per-seed runs are wrapped in a
        single rollup parent run (with a ``log_aggregate`` summary) so the
        MLflow UI shows ONE row per experiment that expands into per-seed
        children. For the single-seed case the run stays flat to avoid an
        empty parent row cluttering the table.
        """
        if self._tracker is None or not seed_results:
            return

        use_parent = len(seed_results) > 1
        parent_ctx = None
        parent_run_id: str | None = None
        if use_parent:
            try:
                parent_ctx = self._tracker.start_parent_run(experiment)
                active_run = parent_ctx.__enter__()
                # ``__enter__`` returns the ActiveRun; capture its id so we can
                # post-process the rollup's duration after all seeds log.
                try:
                    parent_run_id = active_run.info.run_id  # type: ignore[attr-defined]
                except AttributeError:
                    import mlflow  # local import to avoid top-level dep here

                    active = mlflow.active_run()
                    parent_run_id = active.info.run_id if active is not None else None
            except Exception:
                logger.warning("Failed to open MLflow rollup run.", exc_info=True)
                parent_ctx = None

        try:
            for result in seed_results:
                self._log_seed_result(result, experiment, size_tier_map, nested=parent_ctx is not None)

            if parent_ctx is not None:
                try:
                    self._tracker.log_aggregate(seed_results, experiment)
                except Exception:
                    logger.warning("Failed to log MLflow rollup aggregate.", exc_info=True)
        finally:
            if parent_ctx is not None:
                try:
                    parent_ctx.__exit__(None, None, None)
                except Exception:
                    logger.debug("Failed to close MLflow rollup run.", exc_info=True)
                # After the rollup run is terminated, rewrite its end_time so
                # Duration reflects the sum of per-seed wall-clock times rather
                # than the (tiny) logging window.
                if parent_run_id is not None:
                    total_seconds = float(sum(r.total_wall_time() for r in seed_results))
                    try:
                        self._tracker.set_parent_run_duration(parent_run_id, total_seconds)
                    except Exception:
                        logger.debug("Failed to adjust rollup duration.", exc_info=True)

    def _log_seed_result(
        self,
        result: ExperimentResult,
        experiment: Experiment,
        size_tier_map: dict[str, str],
        *,
        nested: bool,
    ) -> None:
        """Log a per-seed result to MLflow (no-op if tracking disabled)."""
        if self._tracker is None:
            return
        try:
            self._tracker.log_run(
                result,
                experiment,
                size_tier=size_tier_map.get(experiment.dataset_name, ""),
                nested=nested,
            )
        except Exception:
            logger.warning("Failed to log run to MLflow.", exc_info=True)

    @staticmethod
    def _is_deterministic_error(exc: BaseException) -> bool:
        """Check if the exception is due to a missing deterministic implementation."""
        if isinstance(exc, PhaseExecutionError):
            exc = exc.original
        return isinstance(exc, RuntimeError) and "does not have a deterministic implementation" in str(exc)

    # Deterministic fallback chain: True → "warn" → False
    _DETERMINISTIC_FALLBACKS: ClassVar[dict[bool | str, bool | str]] = {
        True: "warn",
        "warn": False,
    }

    def _run_single(
        self,
        experiment: Experiment,
        seed: int,
        data_path: Path,
        allowed_phases: set[str],
    ) -> ExperimentResult:
        """Run one ``(experiment, seed)`` with retry and process isolation."""
        benchmark_phases = allowed_phases & _BENCHMARK_PHASE_NAMES
        if benchmark_phases and self.config.isolate_in_subprocess:
            # Do not keep Torch/XPU/NNCF state alive while benchmark_app runs.
            # The benchmark stage gets a fresh Python worker and only reads
            # the exported/optimized model files produced by preparation.
            preparation_phases = allowed_phases - _BENCHMARK_PHASE_NAMES
            prepared = self._run_single_stage(experiment, seed, data_path, preparation_phases)
            if not prepared.success:
                return prepared
            measured = self._run_single_stage(experiment, seed, data_path, benchmark_phases)
            if not measured.success:
                return measured
            return ExperimentResult(
                task=experiment.task,
                model=experiment.model.name,
                dataset=experiment.dataset_name,
                scenario=experiment.scenario.name,
                seed=seed,
                success=True,
                phases=[*prepared.phases, *measured.phases],
            )
        return self._run_single_stage(experiment, seed, data_path, allowed_phases)

    def _run_single_stage(
        self,
        experiment: Experiment,
        seed: int,
        data_path: Path,
        allowed_phases: set[str],
    ) -> ExperimentResult:
        """Run one phase group with retry logic."""
        last_exc: BaseException | None = None
        deterministic: bool | str = self.config.deterministic

        run_fn = self._execute_isolated if self.config.isolate_in_subprocess else self._execute

        max_attempts = max(1, self.config.max_attempts)
        for attempt in range(1, max_attempts + 1):
            try:
                return run_fn(experiment, seed, data_path, allowed_phases, deterministic=deterministic)
            except Exception as exc:  # noqa: PERF203
                last_exc = exc
                if attempt < max_attempts:
                    # If the failure is due to a missing deterministic kernel,
                    # relax the deterministic setting one level for the retry.
                    fallback = self._DETERMINISTIC_FALLBACKS.get(deterministic)
                    if self._is_deterministic_error(exc) and fallback is not None:
                        logger.warning(
                            "  seed=%d attempt %d/%d failed due to non-deterministic op; "
                            "retrying with deterministic=%r: %s",
                            seed,
                            attempt,
                            max_attempts,
                            fallback,
                            exc,
                        )
                        deterministic = fallback
                    else:
                        logger.warning(
                            "  seed=%d attempt %d/%d failed, retrying: %s",
                            seed,
                            attempt,
                            max_attempts,
                            exc,
                        )
                else:
                    logger.exception(
                        "  seed=%d failed after %d attempt(s)",
                        seed,
                        max_attempts,
                    )

        return ExperimentResult.failure(
            task=experiment.task,
            model=experiment.model.name,
            dataset=experiment.dataset_name,
            scenario=experiment.scenario.name,
            seed=seed,
            exc=last_exc,  # type: ignore[arg-type]
        )

    def _execute_isolated(
        self,
        experiment: Experiment,
        seed: int,
        data_path: Path,
        allowed_phases: set[str],
        *,
        deterministic: bool | str | None = None,
    ) -> ExperimentResult:
        """Run ``_execute`` in a spawned child process and return its result.

        The child is a fresh Python interpreter. When it exits, the OS
        reclaims every byte of its memory — DataLoader workers, datumaro
        dm_subset caches, Lightning/MLflow buffers, CUDA context, mmap'd
        weight tensors, everything. This is the only watertight defense
        against the slow host-memory accumulation that otherwise OOM-kills
        the benchmark after ~30 experiments.

        Any exception raised inside the child is reconstructed here as a
        ``RuntimeError`` carrying the child's traceback, so the outer retry
        logic in :py:meth:`_run_single` can still do its thing.
        """
        if deterministic is None:
            deterministic = self.config.deterministic

        # Use a dedicated temp file for the result rather than a Queue/Pipe
        # because some children segfault at shutdown (e.g. after CUDA/NNCF
        # misbehaves) and would leave a Pipe hanging. A file can be inspected
        # even if the child died during final cleanup.
        import contextlib
        import tempfile
        from pathlib import Path as _Path

        with tempfile.NamedTemporaryFile(
            prefix=f"getitune-seed-{experiment.model.name}-{seed}-",
            suffix=".pkl",
            delete=False,
        ) as tmp:
            result_path = tmp.name
        result_file = _Path(result_path)

        try:
            payload = pickle.dumps(
                (self.config, experiment, seed, data_path, allowed_phases, deterministic),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        except Exception:
            logger.exception("Failed to pickle subprocess payload; falling back to in-process execution.")
            with contextlib.suppress(FileNotFoundError):
                result_file.unlink()
            return self._execute(experiment, seed, data_path, allowed_phases, deterministic=deterministic)

        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_subprocess_worker,
            args=(payload, result_path),
            name=f"getitune-{experiment.model.name}-s{seed}",
        )
        proc.start()
        proc.join(self.config.subprocess_timeout)

        if proc.is_alive():
            logger.error(
                "  seed=%d subprocess exceeded timeout (%ss); terminating.",
                seed,
                self.config.subprocess_timeout,
            )
            proc.terminate()
            proc.join(10)
            if proc.is_alive():
                proc.kill()
                proc.join()
            with contextlib.suppress(FileNotFoundError):
                result_file.unlink()
            msg = f"Subprocess timeout after {self.config.subprocess_timeout}s"
            raise TimeoutError(msg)

        exitcode = proc.exitcode
        try:
            with result_file.open("rb") as fh:
                outcome = pickle.load(fh)  # noqa: S301  # nosec B301 - file is written by our own child process to a parent-controlled path
        except (FileNotFoundError, EOFError, pickle.UnpicklingError) as exc:
            # Child died before writing a result — almost always OOM-killed
            # (exitcode = -9 / SIGKILL) or segfault. Surface it as a real
            # exception so _run_single's retry can kick in.
            phase_file = self.config.output_root / experiment.run_id / str(seed) / "current_phase"
            phase = phase_file.read_text(encoding="utf-8").strip() if phase_file.exists() else "unknown"
            msg = f"Subprocess for seed={seed} produced no result (exitcode={exitcode}, phase={phase}): {exc}"
            raise RuntimeError(msg) from exc
        finally:
            with contextlib.suppress(FileNotFoundError):
                result_file.unlink()

        status, payload_out = outcome
        if status == "ok":
            return payload_out  # type: ignore[return-value]

        # status == "error" — re-raise in the parent so the retry logic sees it.
        exc_type_name, exc_msg, tb_str, phase = payload_out  # type: ignore[misc]
        err_msg = f"{exc_type_name}: {exc_msg}\n--- child traceback ---\n{tb_str}"
        if phase is not None:
            # Rebuild a PhaseExecutionError so the failed stage survives the
            # process boundary and reaches ExperimentResult.failure().
            raise PhaseExecutionError(phase, RuntimeError(err_msg))
        raise RuntimeError(err_msg)

    def _execute(
        self,
        experiment: Experiment,
        seed: int,
        data_path: Path,
        allowed_phases: set[str],
        *,
        deterministic: bool | str | None = None,
    ) -> ExperimentResult:
        """Actually run the phases for one seed."""
        if deterministic is None:
            deterministic = self.config.deterministic

        seed_dir = self.config.output_root / experiment.run_id / str(seed)

        # Resume check
        skip, resume_from = detect_resume_point(seed_dir, allowed_phases)
        if skip:
            logger.info("  seed=%d — all phases complete, skipping.", seed)
            return ExperimentResult(
                task=experiment.task,
                model=experiment.model.name,
                dataset=experiment.dataset_name,
                scenario=experiment.scenario.name,
                seed=seed,
                success=True,
                phases=[],
            )

        # Merge scenario overrides with ad-hoc CLI overrides (CLI wins).
        merged_overrides = {**experiment.scenario.overrides}
        merged_overrides.update(self.config.ad_hoc_overrides)

        merged_train_kwargs = {**experiment.scenario.train_kwargs}
        merged_train_kwargs.update(self.config.ad_hoc_train_kwargs)

        executor = ExperimentExecutor(
            recipe_path=experiment.recipe_path,
            data_path=data_path,
            work_dir=seed_dir,
            accelerator=self.config.accelerator,
            scenario_overrides=merged_overrides,
            train_kwargs=merged_train_kwargs,
            seed=seed,
            deterministic=deterministic,
            max_epochs=self.config.max_epochs,
            benchmark_app=self.config.benchmark_app,
            openvino_device=self.config.openvino_device,
            training_device_name=self.config.training_device_name,
            openvino_device_name=self.config.openvino_device_name,
            task=experiment.task,
            model_name=experiment.model.name,
            dataset_name=experiment.dataset_name,
            scenario_name=experiment.scenario.name,
            performance_benchmark=self.config.enable_openvino_benchmark,
        )

        # Determine which phases to run (respecting resume point)
        resuming = resume_from is not None
        phase_results: list[PhaseResult] = []

        for phase_name, method_name in _PHASE_CHAIN:
            if phase_name not in allowed_phases:
                continue
            if resuming and phase_name != resume_from:
                # Still skipping phases until we reach the resume point
                continue
            resuming = False  # From here on, execute everything

            logger.info("  seed=%d  phase=%s", seed, phase_name)
            phase_file = seed_dir / "current_phase"
            phase_file.parent.mkdir(parents=True, exist_ok=True)
            phase_file.write_text(phase_name, encoding="utf-8")
            method = getattr(executor, method_name)
            try:
                result = method()
            except Exception as exc:
                # Tag the failure with the phase that raised it so downstream
                # reporting can show *where* the run failed (train/export/…).
                raise PhaseExecutionError(phase_name, exc) from exc
            finally:
                # Per-phase cleanup: reclaim loky workers spawned by NNCF,
                # flush PyTorch's CUDA cache, and collect cycle garbage so
                # the next phase starts from a clean slate.
                _cleanup_resources()
                phase_file.unlink(missing_ok=True)
            phase_results.extend(result if isinstance(result, list) else [result])

        # Drop the executor (and the heavy engine/trainer state it holds)
        # before returning so the next seed does not inherit its memory.
        del executor
        _cleanup_resources()

        return ExperimentResult(
            task=experiment.task,
            model=experiment.model.name,
            dataset=experiment.dataset_name,
            scenario=experiment.scenario.name,
            seed=seed,
            success=True,
            phases=phase_results,
        )
