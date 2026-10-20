# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Preprocessing throughput harness: detection control vs instance segmentation."""

from __future__ import annotations

import json
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from datumaro.experimental.export_import import import_dataset
from datumaro.experimental.fields import Subset
from tests.benchmarks.instrument import (
    instrument_cpu_transforms,
    instrument_pipeline,
    time_schema_conversion,
)
from tests.benchmarks.rss import PeakRSSSampler
from tests.benchmarks.timing import StageTimer
from torch.utils.data import DataLoader

from getitune.config.data import SubsetConfig
from getitune.data.factory import DatasetFactory
from getitune.types import TaskType


@dataclass
class RunResult:
    """Outcome of one harness run."""

    task: str
    num_workers: int
    batch_size: int
    num_batches: int
    num_samples: int
    wall_s: float
    samples_per_s: float
    peak_total_rss_mb: float
    peak_worker_rss_mb: float
    schema_ms: float = 0.0
    accounted_ms: float = 0.0
    attribution: list[dict[str, Any]] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of wall time explained by the attribution table."""
        return self.accounted_ms / (self.wall_s * 1000.0) if self.wall_s else 0.0


def _subset_config(input_size: tuple[int, int], batch_size: int, num_workers: int) -> SubsetConfig:
    """Build an identical CPU pipeline for both arms of the control experiment.

    Both arms must see the same augmentation list. Any transform present in one
    recipe and not the other lands in the ratio and is indistinguishable from
    mask overhead.
    """
    return SubsetConfig(
        batch_size=batch_size,
        subset_name="train",
        num_workers=num_workers,
        input_size=input_size,
        augmentations_cpu=[
            {
                "class_path": "getitune.data.augmentation.transforms.Resize",
                "init_args": {"size": list(input_size), "keep_aspect_ratio": True},
            },
        ],
        augmentations_gpu=[],
    )


def run_once(
    data_root: Path,
    task: TaskType,
    *,
    input_size: tuple[int, int] = (1024, 1024),
    batch_size: int = 8,
    num_workers: int = 0,
    max_batches: int = 60,
    attribute: bool = True,
) -> RunResult:
    """Measure one (dataset, task) combination.

    Args:
        data_root: Dataset root readable by ``import_dataset``.
        task: ``DETECTION`` for the control arm, ``INSTANCE_SEGMENTATION`` for
            the treatment arm. Both consume the same ``dm_subset``.
        input_size: Target resize, identical across arms.
        batch_size: Dataloader batch size, identical across arms.
        num_workers: Use 0 for per-stage attribution, 8 for throughput.
        max_batches: Batches to consume after warmup.
        attribute: Install per-stage timing hooks. Only meaningful at
            ``num_workers=0``; hooks in worker processes record into a copy of
            the timer that never comes back.

    Returns:
        Throughput, peak RSS and (optionally) the stage attribution table.
    """
    timer = StageTimer()
    dataset_arg = import_dataset(data_root)
    dm_subset = dataset_arg.filter_by_subset(Subset.TRAINING)
    cfg = _subset_config(input_size, batch_size, num_workers)

    with time_schema_conversion(timer):
        vision_dataset = DatasetFactory.create(task=task, dm_subset=dm_subset, cfg_subset=cfg)
    schema_ms = timer.attribution_pop("convert_to_schema") if hasattr(timer, "attribution_pop") else 0.0

    loader = DataLoader(
        vision_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=vision_dataset.collate_fn,
        pin_memory=True,
        persistent_workers=bool(num_workers),
        drop_last=True,
    )

    timer.reset()  # drop construction-time entries; warmup restarts here
    sampler = PeakRSSSampler()

    # Worker processes are spawned on macOS/Windows, which pickles ``collate_fn``
    # by qualified name; a patched module global fails that identity check.
    # The hooks would also record into worker-local timer copies, so they are
    # only installed for the serial, attributing configuration.
    instrument = attribute and num_workers == 0

    seen = 0
    with ExitStack() as stack:
        stack.enter_context(sampler.sampling())
        if instrument:
            stack.enter_context(instrument_pipeline(timer))
            stack.enter_context(instrument_cpu_transforms(vision_dataset.transforms, timer))
        # Warmup is consumed and discarded inside StageTimer; here we only need
        # the wall clock to start after the loader has spun up its workers.
        iterator = iter(loader)
        for _ in range(min(3, max_batches)):
            next(iterator, None)
        start = time.perf_counter()
        for _ in range(max_batches):
            if next(iterator, None) is None:
                break
            seen += 1
        wall_s = time.perf_counter() - start

    num_samples = seen * batch_size
    return RunResult(
        task=task.name,
        num_workers=num_workers,
        batch_size=batch_size,
        num_batches=seen,
        num_samples=num_samples,
        wall_s=round(wall_s, 4),
        samples_per_s=round(num_samples / wall_s, 3) if wall_s else 0.0,
        peak_total_rss_mb=round(sampler.peak_total_mb, 1),
        peak_worker_rss_mb=round(sampler.peak_worker_mb, 1),
        schema_ms=round(schema_ms, 3),
        accounted_ms=round(timer.accounted_ms, 3),
        attribution=timer.attribution() if instrument else [],
    )


def control_experiment(
    data_root: Path,
    *,
    input_size: tuple[int, int] = (1024, 1024),
    batch_size: int = 8,
    num_workers: int = 8,
    max_batches: int = 60,
) -> dict[str, Any]:
    """Run the control: identical images, detection vs instance segmentation.

    The returned ``ratio`` is the mask overhead.
    """
    det = run_once(
        data_root,
        TaskType.DETECTION,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_batches=max_batches,
        attribute=False,
    )
    seg = run_once(
        data_root,
        TaskType.INSTANCE_SEGMENTATION,
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        max_batches=max_batches,
        attribute=False,
    )
    ratio = det.samples_per_s / seg.samples_per_s if seg.samples_per_s else float("inf")
    return {
        "detection": asdict(det),
        "instance_segmentation": asdict(seg),
        "ratio": round(ratio, 3),
        "rss_ratio": round(seg.peak_worker_rss_mb / det.peak_worker_rss_mb, 3) if det.peak_worker_rss_mb else None,
    }


def write_report(payload: dict[str, Any], out_path: Path) -> Path:
    """Persist a benchmark payload as JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
