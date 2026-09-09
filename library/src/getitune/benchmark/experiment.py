# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Experiment executor - thin wrapper around Getitune/OV engines with timing."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess  # nosec B404 - invokes the configured OpenVINO benchmark application
import sys
import threading
import time
import traceback as _traceback
from contextlib import suppress
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from getitune.types.precision import Precision
from getitune.types.task import TaskType

if TYPE_CHECKING:
    import psutil

    from getitune.engine.engine import Engine
    from getitune.metrics import MetricCallable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PhaseExecutionError(Exception):
    """Wraps an exception raised while running a specific benchmark phase.

    Carries the name of the phase (e.g. ``"train"``, ``"export"``,
    ``"optimize"``) so that failure reporting can surface *where* in the
    pipeline the run failed, instead of inferring it heuristically from the
    traceback.
    """

    def __init__(self, phase: str, original: BaseException) -> None:
        self.phase = phase
        self.original = original
        super().__init__(f"Phase '{phase}' failed: {type(original).__name__}: {original}")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    """Metrics collected from a single execution phase."""

    phase: str  # e.g. "train", "test/torch", "test/export", "export", "optimize"
    metrics: dict[str, float] = field(default_factory=dict)
    wall_time: float = 0.0


@dataclass
class ExperimentResult:
    """Aggregated result for one ``(experiment, seed)`` run."""

    task: str
    model: str
    dataset: str
    scenario: str
    seed: int
    success: bool
    phases: list[PhaseResult] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
    failed_phase: str | None = None  # e.g. "train", "export", "optimize"

    def all_metrics(self) -> dict[str, float]:
        """Merge metrics from all phases into a single dict."""
        merged: dict[str, float] = {}
        for phase in self.phases:
            merged.update(phase.metrics)
        return merged

    def total_wall_time(self) -> float:
        """Sum of wall-clock seconds across every phase of this seed run."""
        return float(sum(p.wall_time for p in self.phases))

    @classmethod
    def failure(
        cls,
        *,
        task: str,
        model: str,
        dataset: str,
        scenario: str,
        seed: int,
        exc: BaseException,
        failed_phase: str | None = None,
    ) -> ExperimentResult:
        """Construct a failed result from an exception.

        When *exc* is a :class:`PhaseExecutionError`, the wrapped original
        exception is used for the error message/traceback and its ``phase``
        is recorded as :attr:`failed_phase` (unless *failed_phase* is given
        explicitly, which then takes precedence).
        """
        phase = failed_phase
        if isinstance(exc, PhaseExecutionError):
            if phase is None:
                phase = exc.phase
            exc = exc.original
        tb_str = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
        return cls(
            task=task,
            model=model,
            dataset=dataset,
            scenario=scenario,
            seed=seed,
            success=False,
            phases=[],
            error=f"{type(exc).__name__}: {exc}",
            traceback=tb_str,
            failed_phase=phase,
        )


# ---------------------------------------------------------------------------
# Override resolution
# ---------------------------------------------------------------------------


def resolve_overrides(scenario_overrides: dict[str, Any]) -> dict[str, Any]:
    """Convert scenario overrides into kwargs for ``LightningEngine.from_config()``.

    Complex values (dicts, lists) are JSON-serialized so that jsonargparse can
    parse them on the engine side.
    """
    resolved: dict[str, Any] = {}
    for dotpath, value in scenario_overrides.items():
        if isinstance(value, (dict, list)):
            resolved[dotpath] = json.dumps(value)
        else:
            resolved[dotpath] = value
    return resolved


# ---------------------------------------------------------------------------
# Resume detection
# ---------------------------------------------------------------------------

# Maps (phase_name -> marker file relative to seed dir) for resume checks.
_PHASE_MARKERS: list[tuple[str, str]] = [
    ("train", "train/metrics.csv"),
    ("test/torch", "test/torch/result.json"),
    ("export", "export/exported_model.xml"),
    ("test/export", "test/export/result.json"),
    ("benchmark/export", "benchmark/export/result.json"),
    ("optimize", "optimize/optimized_model.xml"),
    ("test/optimize", "test/optimize/result.json"),
    ("benchmark/optimize", "benchmark/optimize/result.json"),
]


def _training_artifacts_complete(seed_dir: Path) -> bool:
    """Return whether training metrics and the best checkpoint both exist.

    Lightning and Ultralytics both write per-run CSVs under
    ``train/csv/version_*/metrics.csv``; the direct ``train/metrics.csv`` path
    is kept as a legacy fallback for hand-crafted result directories.
    """
    metric_candidates = [
        *sorted(seed_dir.glob("train/csv/version_*/metrics.csv")),
        seed_dir / "train" / "metrics.csv",
    ]
    has_metrics = any(path.exists() and path.stat().st_size > 0 for path in metric_candidates)
    return has_metrics and (seed_dir / "train" / "best_checkpoint.pt").exists()


def _has_measured_artifacts(seed_dir: Path) -> bool:
    """Return whether the seed dir holds performance measurements worth keeping."""
    return (seed_dir / "performance_result.json").exists() or (seed_dir / "benchmark").is_dir()


def detect_resume_point(seed_dir: Path, required_phases: set[str] | None = None) -> tuple[bool, str | None]:
    """Determine whether an experiment can be skipped or partially resumed.

    Returns:
        ``(True, None)`` - all phases complete, skip entirely.
        ``(False, None)`` - start from scratch.
        ``(False, <phase_name>)`` - training done, resume from this phase.
    """
    if not _training_artifacts_complete(seed_dir):
        # Training not done or corrupt -> start over. Never destroy a seed
        # directory that still holds performance measurements: a later worker
        # (e.g. the isolated benchmark stage) must not wipe artifacts produced
        # by the preparation stage.
        if seed_dir.exists() and not _has_measured_artifacts(seed_dir):
            shutil.rmtree(seed_dir)
        return False, None

    checkpoint_exists = (seed_dir / "train" / "best_checkpoint.pt").exists()
    if not checkpoint_exists:
        # metrics exist but checkpoint missing -> corrupt
        shutil.rmtree(seed_dir)
        return False, None

    # Training is complete. Walk only phases requested by this invocation.
    # Callers that do not provide a phase set use the legacy phase chain. This
    # keeps direct resume checks compatible with result directories created
    # before benchmark-app phases existed.
    phases = required_phases or {
        phase_name for phase_name, _ in _PHASE_MARKERS if not phase_name.startswith("benchmark/")
    }
    for phase_name, marker_rel in _PHASE_MARKERS[1:]:  # skip "train"
        if phase_name not in phases:
            continue
        marker = seed_dir / marker_rel
        if not marker.exists():
            return False, phase_name

    return True, None


# ---------------------------------------------------------------------------
# Metric scraping helpers
# ---------------------------------------------------------------------------


def _scrape_csv_metrics(csv_path: Path, prefix: str) -> dict[str, float]:
    """Read a Lightning ``metrics.csv`` and extract key aggregates.

    The prefix is prepended to each metric key (e.g. ``"training:"``).
    The ``epoch`` column is only meaningful for the training phase; for
    inference-only phases (test/export/optimize) Lightning still writes an
    ``epoch=0`` row, so we suppress it to avoid emitting a misleading
    ``<phase>:epoch = 1`` metric.
    """
    if not csv_path.exists():
        return {}
    try:
        raw_metrics = pd.read_csv(csv_path)
    except Exception:
        logger.warning("Could not parse %s", csv_path)
        return {}

    is_training_phase = prefix == "training:"
    metrics: dict[str, float] = {}
    for col in raw_metrics.columns:
        series = raw_metrics[col].dropna()
        if series.empty:
            continue
        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            continue
        # For val/test accuracy metrics, take the max (best epoch / the
        # single logged value). For timing metrics, take the mean (skip
        # the first warmup step).
        if "val/" in col:
            metrics[f"{prefix}{col}"] = float(series.max())
        elif "iter_time" in col:
            trimmed = series.iloc[min(1, len(series) - 1) :]
            metrics[f"{prefix}{col}"] = float(trimmed.mean())
        elif "test/" in col:
            metrics[f"{prefix}{col}"] = float(series.max())
        elif "epoch" in col:
            if not is_training_phase:
                # Inference-only phases don't have a meaningful epoch count.
                continue
            # Lightning records ``epoch`` as a 0-indexed counter, so the max
            # is ``num_epochs - 1``.  Report the human-readable count instead.
            metrics[f"{prefix}{col}"] = float(series.max()) + 1.0
        elif "gpu_mem" in col or "gpu" in col.lower():
            metrics[f"{prefix}{col}"] = float(series.max())
    return metrics


def _find_csv_metrics(csv_dir: Path) -> Path | None:
    """Locate the Lightning CSV logger's ``metrics.csv`` under *csv_dir*.

    Lightning writes CSV files to ``csv/version_N/metrics.csv``, where *N*
    increments on each run.  This helper returns the ``metrics.csv`` inside
    the highest ``version_*`` directory, so it works even when multiple
    training sessions have run (e.g. during resume).

    Falls back to ``csv_dir / "metrics.csv"`` if the ``version_*`` layout
    is not present (for forward compatibility).
    """
    csv_parent = csv_dir / "csv"
    if csv_parent.is_dir():
        # Find all version_* directories and pick the highest number
        version_dirs = sorted(
            (d for d in csv_parent.iterdir() if d.is_dir() and d.name.startswith("version_")),
            key=lambda d: int(d.name.split("_", 1)[1]) if d.name.split("_", 1)[1].isdigit() else -1,
        )
        if version_dirs:
            candidate = version_dirs[-1] / "metrics.csv"
            if candidate.exists():
                return candidate
    # Fallback: direct metrics.csv (e.g. train/metrics.csv)
    direct = csv_dir / "metrics.csv"
    if direct.exists():
        return direct
    return None


def _get_peak_gpu_memory_mb() -> float:
    """Best-effort peak accelerator memory reading in MB (returns 0.0 if unavailable).

    Covers both CUDA (NVIDIA) and XPU (Intel) devices. XPU is first-class, so
    we must not assume a CUDA-only environment here.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.max_memory_allocated() / (1024 * 1024)
    except Exception:
        logger.debug("Could not read peak accelerator memory.", exc_info=True)
    return 0.0


def _reset_peak_gpu_memory() -> None:
    """Best-effort reset of the accelerator's peak-memory counter (no-op if unavailable).

    Scopes the next :func:`_get_peak_gpu_memory_mb` reading to the phase that
    follows this call, rather than accumulating since the last experiment-level
    reset.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.reset_peak_memory_stats()
    except Exception:
        logger.debug("Could not reset peak accelerator memory.", exc_info=True)


def _safe_rss(process: psutil.Process) -> int:
    """Return *process*' RSS in bytes, or ``0`` if it has already exited."""
    try:
        return process.memory_info().rss
    except Exception:
        return 0


class _PeakRamSampler:
    """Background sampler that tracks peak host RAM (RSS) during a ``with`` block.

    Polls this process' RSS plus every child process' RSS (e.g. DataLoader
    workers) on a fixed interval and keeps the running maximum. Degrades to a
    no-op (``peak_mb`` stays ``0.0``) when ``psutil`` isn't installed, since it
    is only declared under the optional ``benchmark`` extra.
    """

    _POLL_INTERVAL_S = 0.2
    _JOIN_TIMEOUT_S = 2.0

    def __init__(self) -> None:
        self._peak_mb = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: psutil.Process | None = None

    def __enter__(self) -> _PeakRamSampler:
        try:
            import psutil

            self._process = psutil.Process()
        except Exception:
            logger.debug("psutil unavailable; RAM sampling disabled (install the 'benchmark' extra).")
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._JOIN_TIMEOUT_S)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(self._POLL_INTERVAL_S)
        self._sample()  # capture a final reading right before exit

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            procs = [self._process, *self._process.children(recursive=True)]
            total_bytes = sum(_safe_rss(proc) for proc in procs)
            self._peak_mb = max(self._peak_mb, total_bytes / (1024 * 1024))
        except Exception:
            logger.debug("RAM sampling tick failed (ignored).", exc_info=True)

    @property
    def peak_mb(self) -> float:
        """Peak observed RAM (main process + children) in MB, ``0.0`` if unavailable."""
        return self._peak_mb


def _count_test_samples(engine: Engine) -> int:
    """Return the number of test samples for *engine*'s datamodule (>= 1).

    The engine's ``datamodule`` may be a :class:`DataModule` (exposing
    ``subsets``) or a filesystem path (Ultralytics data-root mode). Falls back
    to ``1`` when the count cannot be determined so latency math stays safe.
    """
    datamodule = engine.datamodule
    subsets = getattr(datamodule, "subsets", None)
    if isinstance(subsets, dict):
        return max(len(subsets.get("test", [])), 1)
    return 1


# ---------------------------------------------------------------------------
# Ultralytics backend helpers
# ---------------------------------------------------------------------------


def _recipe_backend(recipe_path: Path) -> tuple[str, TaskType | None]:
    """Inspect a recipe and return ``(backend, task_type)``.

    ``backend`` is ``"ultralytics"`` when the recipe declares
    ``backend: ultralytics``, otherwise ``"lightning"``.  ``task_type`` is the
    parsed :class:`TaskType` for ultralytics recipes (read from the ``task``
    field) and ``None`` for the Lightning path.
    """
    try:
        with recipe_path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        msg = f"Could not load recipe {recipe_path}: {exc}"
        raise ValueError(msg) from exc
    if isinstance(raw, dict) and raw.get("backend") == "ultralytics":
        task_raw = raw.get("task")
        task_type = TaskType(task_raw) if task_raw else None
        return "ultralytics", task_type
    return "lightning", None


def _ultralytics_torch_metric(task_type: TaskType | None) -> MetricCallable | None:
    """Return the torchmetrics callable Lightning uses for *task_type*.

    Driving the Ultralytics engine's torchmetrics evaluation path with the same
    callable keeps the produced metric names (e.g. ``test/map_50``,
    ``test/f1-score``) comparable across backends.
    """
    from getitune.metrics.fmeasure import (
        MaskRLEMeanAPFMeasureCallable,
        MeanAveragePrecisionFMeasureCallable,
    )

    return {
        TaskType.DETECTION: MeanAveragePrecisionFMeasureCallable,
        TaskType.INSTANCE_SEGMENTATION: MaskRLEMeanAPFMeasureCallable,
    }.get(task_type)  # type: ignore[arg-type]


def _write_phase_metrics_csv(work_dir: Path, metrics: dict[str, Any] | None) -> None:
    """Persist a flat metric dict as ``<work_dir>/csv/version_0/metrics.csv``.

    The Ultralytics engine returns metrics from ``test()`` but, unlike
    Lightning's ``CSVLogger``, does not write them to disk.  Writing them to the
    same path Lightning uses lets the existing scraping logic find them.
    """
    if not metrics:
        return
    scalar_metrics = {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) or hasattr(value, "item")
    }
    if not scalar_metrics:
        return
    csv_dir = work_dir / "csv" / "version_0"
    csv_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([scalar_metrics]).to_csv(csv_dir / "metrics.csv", index=False)


def _parse_benchmark_value(value: object) -> float | None:
    """Extract a float from benchmark-app's string-valued JSON report."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value.split()[0])
    except (IndexError, ValueError):
        return None


def _parse_benchmark_report(report_path: Path, *, prefix: str) -> dict[str, float]:
    """Parse OpenVINO benchmark-app JSON execution results into flat metrics."""
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        execution = raw.get("execution_results", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        logger.warning("Could not parse benchmark-app report %s", report_path)
        return {}

    aliases = {
        "throughput": "fps",
        "latency (ms)": "latency_ms",
        "latency (50 percentile) (ms)": "latency_ms",
        "avg latency": "avg_latency_ms",
        "total execution time (ms)": "duration_ms",
        "total number of iterations": "iterations",
    }
    metrics: dict[str, float] = {}
    for key, metric_name in aliases.items():
        value = _parse_benchmark_value(execution.get(key))
        if value is not None:
            metrics[f"{prefix}{metric_name}"] = value
    return metrics


def _parse_benchmark_setup(report_path: Path) -> dict[str, object]:
    """Return benchmark-app's effective runtime configuration."""
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        setup = raw.get("configuration_setup", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    return setup if isinstance(setup, dict) else {}


def _benchmark_report_error(report_path: Path) -> str | None:
    """Return benchmark-app's reported execution error, if any."""
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        error = raw.get("execution_results", {}).get("error")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return str(error) if error else None


def _default_benchmark_app() -> str:
    """Return the benchmark-app executable for the current Python environment."""
    return str(Path(sys.executable).with_name("benchmark_app"))


def _benchmark_device(accelerator: str) -> str:
    """Map a Geti accelerator label to an OpenVINO device label."""
    return {"cpu": "CPU", "gpu": "GPU", "xpu": "GPU", "mps": "CPU"}.get(accelerator.lower(), accelerator)


def _benchmark_data_shape(model: Path) -> str | None:
    """Return a concrete input shape for benchmark-app dynamic inputs."""
    try:
        import openvino as ov

        input_port = ov.Core().read_model(model).inputs[0]
        dimensions = [str(dim.get_length() if dim.is_static else 1) for dim in input_port.partial_shape]
    except Exception:
        logger.warning("Could not determine a concrete input shape for %s", model, exc_info=True)
        return None
    return "[" + ",".join(dimensions) + "]"


def _as_positive_int(value: object) -> int | None:
    """Convert a batch-size value to a positive integer when possible."""
    try:
        converted = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return converted if converted > 0 else None


def _validate_fp16_model(model: Path) -> None:
    """Verify that an exported IR contains FP16 constants."""
    import openvino as ov

    ov_model = ov.Core().read_model(model)
    has_fp16_weights = any(
        operation.get_type_name() == "Constant" and operation.get_output_element_type(0) == ov.Type.f16
        for operation in ov_model.get_ops()
    )
    if not has_fp16_weights:
        msg = f"Exported model does not contain FP16 weights: {model}"
        raise RuntimeError(msg)


def _package_version(name: str) -> str:
    """Return an installed package version, failing clearly when unavailable."""
    try:
        return version(name)
    except PackageNotFoundError as exc:
        msg = f"Required package version metadata is unavailable: {name}"
        raise RuntimeError(msg) from exc


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class ExperimentExecutor:
    """Run train / test / export / optimize for a single experiment + seed.

    The executor is intentionally stateless with respect to tracking -- it
    returns structured :class:`PhaseResult` objects that the runner can
    forward to any tracking backend.
    """

    def __init__(
        self,
        *,
        recipe_path: Path,
        data_path: Path,
        work_dir: Path,
        accelerator: str = "gpu",
        scenario_overrides: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        seed: int = 0,
        deterministic: bool | str = True,
        max_epochs: int | None = None,
        benchmark_app: str | None = None,
        openvino_device: str | None = None,
        training_device_name: str | None = None,
        openvino_device_name: str | None = None,
        task: str = "",
        model_name: str = "",
        dataset_name: str = "",
        scenario_name: str = "default",
        performance_benchmark: bool = False,
    ) -> None:
        self.recipe_path = recipe_path
        self.data_path = data_path
        self.work_dir = work_dir
        self.accelerator = accelerator
        self.scenario_overrides = scenario_overrides or {}
        self.extra_train_kwargs = train_kwargs or {}
        self.seed = seed
        self.deterministic = deterministic
        self.max_epochs = max_epochs
        self.benchmark_app = benchmark_app or _default_benchmark_app()
        self.openvino_device = openvino_device or _benchmark_device(accelerator)
        self.training_device_name = training_device_name
        self.openvino_device_name = openvino_device_name
        self.task = task
        self.model_name = model_name or recipe_path.stem
        self.dataset_name = dataset_name or data_path.name
        self.scenario_name = scenario_name
        self.performance_benchmark = performance_benchmark
        self._backend, self._task_type = _recipe_backend(recipe_path)

    @property
    def is_ultralytics(self) -> bool:
        """Whether this experiment's recipe uses the Ultralytics backend."""
        return self._backend == "ultralytics"

    @property
    def _checkpoint_name(self) -> str:
        """Trained-checkpoint filename written by the engine after training."""
        return "best_checkpoint.pt"

    def _build_torch_engine(self, work_dir: Path) -> Engine:
        """Build the torch-side engine (Lightning or Ultralytics) for *work_dir*.

        The Ultralytics path mirrors the application's getitune trainer: it
        builds a model + datamodule from the recipe and dispatches them through
        the library's ``create_engine`` factory.
        """
        if self.is_ultralytics:
            from getitune.backend.ultralytics.tools.configurator import Configurator
            from getitune.engine import create_engine

            configurator = Configurator(
                data=self.data_path,
                model=self.recipe_path,
                task=self._task_type,
            )
            if self.scenario_overrides:
                try:
                    configurator.apply_overrides(self.scenario_overrides)
                except (KeyError, ValueError, TypeError):
                    logger.warning(
                        "Could not apply scenario overrides %s to Ultralytics recipe %s; ignoring.",
                        self.scenario_overrides,
                        self.recipe_path,
                    )
            datamodule = configurator.build_datamodule()
            model: Any = configurator.create_model(datamodule.label_info)
            return create_engine(
                model=model,
                data=datamodule,
                work_dir=work_dir,
                device=self.accelerator,
                train_args=configurator.training,
                export_args={
                    "confidence_threshold": configurator.export.get("confidence_threshold", 0.25),
                    "iou_threshold": configurator.export.get("iou_threshold", 0.5),
                },
            )

        from getitune.backend.lightning.engine import LightningEngine

        overrides = resolve_overrides(self.scenario_overrides)
        return LightningEngine.from_config(
            config_path=self.recipe_path,
            data=self.data_path,
            work_dir=work_dir,
            device=self.accelerator,
            **overrides,
        )

    # -- phases ------------------------------------------------------------

    def train(self) -> PhaseResult:
        """Train the model and return scraped metrics."""
        engine = self._build_torch_engine(self.work_dir / "train")

        kwargs: dict[str, Any] = {"seed": self.seed}
        # The Ultralytics trainer expects a boolean ``deterministic``; coerce the
        # Lightning-style "warn" sentinel away for that backend.
        deterministic = self.deterministic
        if self.is_ultralytics and not isinstance(deterministic, bool):
            deterministic = False
        kwargs["deterministic"] = deterministic
        if self.max_epochs is not None and self.max_epochs > 0:
            kwargs["max_epochs"] = self.max_epochs
        kwargs.update(self.extra_train_kwargs)

        _reset_peak_gpu_memory()
        start = time.monotonic()
        with _PeakRamSampler() as ram_sampler:
            engine.train(**kwargs)
        wall = time.monotonic() - start

        # Scrape metrics from the CSV that the engine writes
        csv_path = _find_csv_metrics(self.work_dir / "train")
        csv_metrics = _scrape_csv_metrics(csv_path, prefix="training:") if csv_path else {}
        csv_metrics["training:e2e_time"] = wall
        csv_metrics["training:gpu_mem"] = _get_peak_gpu_memory_mb()
        csv_metrics["training:ram_mem"] = ram_sampler.peak_mb

        if self.performance_benchmark:
            from getitune.benchmark.hardware import get_training_device_name

            train_batch_size = self._effective_training_batch_size(engine)
            if train_batch_size is None:
                msg = f"Could not determine effective training batch size for {self.model_name}."
                raise RuntimeError(msg)
            metadata = {
                "schema_version": 1,
                "task": self.task,
                "model": self.model_name,
                "dataset": self.dataset_name,
                "scenario": self.scenario_name,
                "seed": self.seed,
                "training_device": self.training_device_name or get_training_device_name(self.accelerator),
                "training_batch_size": train_batch_size,
                "gpu_memory_mb": csv_metrics.get("training:gpu_mem", 0.0),
                "ram_memory_mb": csv_metrics.get("training:ram_mem", 0.0),
                "software": self._software_versions(),
            }
            self._write_performance_result(metadata)

        del engine
        return PhaseResult(phase="train", metrics=csv_metrics, wall_time=wall)

    def _effective_training_batch_size(self, engine: Engine) -> int | None:
        """Return the actual or configured training batch size."""
        datamodule = engine.datamodule
        train_subset = getattr(datamodule, "train_subset", None)
        train_batch_size = _as_positive_int(getattr(train_subset, "batch_size", None))
        if self.is_ultralytics:
            ultralytics_engine: Any = engine
            trainer = getattr(ultralytics_engine.model.yolo, "trainer", None)
            train_batch_size = _as_positive_int(getattr(trainer, "batch_size", None))
            if train_batch_size is None:
                train_batch_size = _as_positive_int(getattr(getattr(trainer, "args", None), "batch", None))
            if train_batch_size is None:
                train_batch_size = _as_positive_int(getattr(ultralytics_engine, "_train_args", {}).get("batch"))
            if train_batch_size is None:
                train_batch_size = _as_positive_int(getattr(train_subset, "batch_size", None))
        return train_batch_size

    def test_torch(self) -> PhaseResult:
        """Test the PyTorch checkpoint and return metrics."""
        engine = self._build_torch_engine(self.work_dir / "test" / "torch")
        ckpt = self.work_dir / "train" / self._checkpoint_name

        test_kwargs: dict[str, Any] = {}
        if self.is_ultralytics:
            # Drive the shared torchmetrics evaluation path so metric names match
            # the Lightning backend (e.g. test/map_50, test/f1-score).
            metric_callable = _ultralytics_torch_metric(self._task_type)
            if metric_callable is not None:
                test_kwargs["metric"] = metric_callable

        _reset_peak_gpu_memory()
        start = time.monotonic()
        with _PeakRamSampler() as ram_sampler:
            metrics = engine.test(checkpoint=ckpt, **test_kwargs)
        wall = time.monotonic() - start

        # The Ultralytics engine returns metrics without writing a metrics.csv;
        # persist them where the scraper looks so both backends behave the same.
        if self.is_ultralytics:
            _write_phase_metrics_csv(Path(engine.work_dir), metrics)

        num_samples = _count_test_samples(engine)
        latency = wall / num_samples

        csv_path = _find_csv_metrics(Path(engine.work_dir))
        csv_metrics = _scrape_csv_metrics(csv_path, prefix="torch:") if csv_path else {}
        csv_metrics["torch:test/e2e_time"] = wall
        csv_metrics["torch:test/latency"] = latency
        csv_metrics["torch:test/gpu_mem"] = _get_peak_gpu_memory_mb()
        csv_metrics["torch:test/ram_mem"] = ram_sampler.peak_mb

        # Write a marker for resume detection
        result_json = self.work_dir / "test" / "torch" / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(csv_metrics, indent=2))

        del engine
        return PhaseResult(phase="test/torch", metrics=csv_metrics, wall_time=wall)

    def export(self) -> PhaseResult:
        """Export the trained model to OpenVINO IR."""
        engine = self._build_torch_engine(self.work_dir / "export")
        ckpt = self.work_dir / "train" / self._checkpoint_name

        start = time.monotonic()
        export_kwargs = {"export_precision": Precision.FP16} if self.performance_benchmark else {}
        exported = Path(engine.export(checkpoint=ckpt, **export_kwargs))
        if self.performance_benchmark:
            _validate_fp16_model(exported)
        wall = time.monotonic() - start

        del engine
        return PhaseResult(phase="export", metrics={"export:e2e_time": wall}, wall_time=wall)

    def test_export(self) -> PhaseResult:
        """Test the exported OpenVINO model and return metrics."""
        from getitune.backend.openvino.engine import OVEngine

        exported = self._find_exported_model()
        engine = OVEngine(
            work_dir=self.work_dir / "test" / "export",
            data=self.data_path,
            model=exported,
        )

        start = time.monotonic()
        engine.test(checkpoint=exported)
        wall = time.monotonic() - start

        num_samples = _count_test_samples(engine)
        latency = wall / num_samples

        csv_path = _find_csv_metrics(Path(engine.work_dir))
        csv_metrics = _scrape_csv_metrics(csv_path, prefix="export:") if csv_path else {}
        csv_metrics["export:test/e2e_time"] = wall
        csv_metrics["export:test/latency"] = latency

        result_json = self.work_dir / "test" / "export" / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(csv_metrics, indent=2))

        del engine
        return PhaseResult(phase="test/export", metrics=csv_metrics, wall_time=wall)

    def benchmark_export(self) -> PhaseResult:
        """Measure FP16 throughput and latency and persist structured results."""
        results = self._benchmark_model(self._find_exported_model(), "export")
        self._update_performance_result("fp16", results)
        self._write_benchmark_marker("export", results)
        return results

    def optimize(self) -> PhaseResult:
        """Optimize the exported model with NNCF/POT."""
        from getitune.backend.openvino.engine import OVEngine

        exported = self._find_exported_model()
        engine = OVEngine(
            work_dir=self.work_dir / "optimize",
            data=self.data_path,
            model=exported,
        )

        start = time.monotonic()
        engine.optimize(checkpoint=exported)
        wall = time.monotonic() - start

        del engine
        return PhaseResult(phase="optimize", metrics={"optimize:e2e_time": wall}, wall_time=wall)

    def test_optimize(self) -> PhaseResult:
        """Test the optimized model and return metrics."""
        from getitune.backend.openvino.engine import OVEngine

        optimized = self.work_dir / "optimize" / "optimized_model.xml"
        if not optimized.exists():
            msg = f"Optimized model not found: {optimized}"
            raise FileNotFoundError(msg)

        engine = OVEngine(
            work_dir=self.work_dir / "test" / "optimize",
            data=self.data_path,
            model=optimized,
        )

        start = time.monotonic()
        engine.test(checkpoint=optimized)
        wall = time.monotonic() - start

        num_samples = _count_test_samples(engine)
        latency = wall / num_samples

        csv_path = _find_csv_metrics(Path(engine.work_dir))
        csv_metrics = _scrape_csv_metrics(csv_path, prefix="optimize:") if csv_path else {}
        csv_metrics["optimize:test/e2e_time"] = wall
        csv_metrics["optimize:test/latency"] = latency

        result_json = self.work_dir / "test" / "optimize" / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(csv_metrics, indent=2))

        del engine
        return PhaseResult(phase="test/optimize", metrics=csv_metrics, wall_time=wall)

    def benchmark_optimize(self) -> PhaseResult:
        """Measure INT8 throughput and latency and persist structured results."""
        results = self._benchmark_model(self._optimized_model(), "optimize")
        self._update_performance_result("int8", results)
        self._write_benchmark_marker("optimize", results)
        return results

    def _optimized_model(self) -> Path:
        """Return the optimized INT8 model path."""
        optimized = self.work_dir / "optimize" / "optimized_model.xml"
        if not optimized.exists():
            msg = f"Optimized model not found: {optimized}"
            raise FileNotFoundError(msg)
        return optimized

    def _benchmark_model(self, model: Path, model_kind: str) -> PhaseResult:
        """Run throughput and latency measurements for one model variant."""
        combined_metrics: dict[str, float] = {}
        wall = 0.0
        for hint in ("throughput", "latency"):
            metrics, elapsed = self._run_benchmark_app(model, model_kind, hint)
            combined_metrics.update(metrics)
            wall += elapsed
        return PhaseResult(phase=f"benchmark/{model_kind}", metrics=combined_metrics, wall_time=wall)

    def _write_benchmark_marker(self, model_kind: str, phase: PhaseResult) -> None:
        """Write a resume marker after structured result persistence succeeds."""
        marker = self.work_dir / "benchmark" / model_kind / "result.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(phase.metrics, indent=2), encoding="utf-8")

    def _run_benchmark_app(self, model: Path, model_kind: str, hint: str) -> tuple[dict[str, float], float]:
        """Run and validate one benchmark-app mode."""
        output_dir = self.work_dir / "benchmark" / model_kind / hint
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "benchmark_report.json"
        data_shape = _benchmark_data_shape(model)
        command = [
            self.benchmark_app,
            "-m",
            str(model),
            "-d",
            self.openvino_device,
            "-hint",
            hint,
            "-report_type",
            "no_counters",
            "-report_folder",
            str(output_dir),
            "-json_stats",
        ]
        if data_shape is not None:
            # ``-data_shape`` supplies generated input data, while ``-shape``
            # concretizes dynamic model inputs before device compilation.
            command.extend(["-shape", data_shape])
            command.extend(["-data_shape", data_shape])
        if hint == "latency":
            command.extend(["-b", "1", "-nireq", "1"])
        report_path.unlink(missing_ok=True)
        start = time.monotonic()
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, env=os.environ.copy())  # noqa: S603
        except OSError as exc:
            msg = f"benchmark_app could not start for {model_kind}/{hint}: {exc}"
            raise RuntimeError(msg) from exc
        wall = time.monotonic() - start
        log = f"$ {' '.join(command)}\n\n{completed.stdout}\n{completed.stderr}"
        (output_dir / "benchmark_app.log").write_text(log, encoding="utf-8")
        report_error = _benchmark_report_error(report_path)
        if report_error is not None:
            msg = f"benchmark_app failed for {model_kind}/{hint}: {report_error}"
            raise RuntimeError(msg)
        metrics = _parse_benchmark_report(report_path, prefix=f"{model_kind}:{hint}:")
        setup = _parse_benchmark_setup(report_path)
        batch_value = setup.get("batch size")
        if isinstance(batch_value, (int, float, str)):
            with suppress(ValueError, TypeError):
                metrics[f"{model_kind}:{hint}:batch_size"] = float(batch_value)
        required = {f"{model_kind}:{hint}:fps", f"{model_kind}:{hint}:latency_ms", f"{model_kind}:{hint}:batch_size"}
        if not required <= metrics.keys():
            msg = f"benchmark_app produced an incomplete report for {model_kind}/{hint}: {report_path}"
            raise RuntimeError(msg)
        if hint == "latency" and metrics[f"{model_kind}:{hint}:batch_size"] != 1:
            msg = f"benchmark_app latency batch must be 1, got {metrics[f'{model_kind}:{hint}:batch_size']}"
            raise RuntimeError(msg)
        if completed.returncode != 0:
            logger.warning("benchmark_app exited with %d after writing a complete report.", completed.returncode)
        (output_dir / "result.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics, wall

    def _update_performance_result(self, precision: str, phase: PhaseResult) -> None:
        """Update the canonical per-seed performance result."""
        from getitune.benchmark.hardware import get_openvino_device_name

        output = self.work_dir / "performance_result.json"
        try:
            current = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            msg = f"Missing canonical performance result: {output}"
            raise RuntimeError(msg) from exc
        current.update(
            {
                "openvino_device": self.openvino_device_name or get_openvino_device_name(self.openvino_device),
                "openvino_target": self.openvino_device,
            }
        )
        prefix = "export" if precision == "fp16" else "optimize"
        current[precision] = {
            "precision": "FP16" if precision == "fp16" else "INT8",
            "inference_batch_size": int(phase.metrics[f"{prefix}:throughput:batch_size"]),
            "throughput_fps": phase.metrics[f"{prefix}:throughput:fps"],
            "latency_ms": phase.metrics[f"{prefix}:latency:latency_ms"],
        }
        self._write_performance_result(current)

    def _write_performance_result(self, result: dict[str, Any]) -> None:
        """Atomically write the canonical performance result."""
        output = self.work_dir / "performance_result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(output)

    @staticmethod
    def _git_sha() -> str:
        """Return the current Git revision for result provenance."""
        from getitune.benchmark.tracking import get_git_sha

        return get_git_sha()

    @staticmethod
    def _software_versions() -> dict[str, str]:
        """Return the software versions needed to reproduce measurements."""
        return {
            "python": platform.python_version(),
            "getitune": _package_version("getitune"),
            "torch": _package_version("torch"),
            "openvino": _package_version("openvino"),
            "nncf": _package_version("nncf"),
        }

    # -- helpers -----------------------------------------------------------

    def _find_exported_model(self) -> Path:
        """Locate the exported OpenVINO IR XML file."""
        candidates = [
            self.work_dir / "export" / "exported_model.xml",
            self.work_dir / ".latest" / "export" / "exported_model_decoder.xml",
        ]
        for p in candidates:
            if p.exists():
                return p
        msg = f"Exported model not found in any of: {candidates}"
        raise FileNotFoundError(msg)
