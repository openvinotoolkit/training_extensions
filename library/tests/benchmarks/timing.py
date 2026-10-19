# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Stage-level timing primitives for the preprocessing benchmark."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import torch

from getitune.utils.device import is_xpu_available


def sync_device(device: str | torch.device | None = None) -> None:
    """Block until all queued accelerator work has completed.

    Stages like GPU augmentation & mask target generation are asynchronous.
    Without this, ``perf_counter`` around them reports close to zero and the
    cost silently reappears in whichever stage happens to synchronise next.
    """
    dev = str(device) if device is not None else ""
    if dev.startswith("cuda") or (not dev and torch.cuda.is_available()):
        torch.cuda.synchronize()
    elif dev.startswith("xpu") or (not dev and is_xpu_available()):
        torch.xpu.synchronize()


@dataclass
class StageStats:
    """Aggregated timings for one stage, in nanoseconds."""

    name: str
    samples: list[int] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of recorded invocations."""
        return len(self.samples)

    @property
    def total_ms(self) -> float:
        """Total wall time across all invocations, in milliseconds."""
        return sum(self.samples) / 1e6

    @property
    def mean_ms(self) -> float:
        """Mean wall time per invocation, in milliseconds."""
        return self.total_ms / self.count if self.count else 0.0

    @property
    def median_ms(self) -> float:
        """Median wall time per invocation, in milliseconds."""
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        return ordered[len(ordered) // 2] / 1e6


class StageTimer:
    """Process-local accumulator of per-stage timings.

    Deliberately not multiprocess-aware. Per-stage attribution is collected
    with ``num_workers=0`` so the numbers sum cleanly; end-to-end throughput is
    measured separately at the real worker count.
    """

    def __init__(self, device: str | torch.device | None = None) -> None:
        self._stages: dict[str, StageStats] = {}
        self._warmup: dict[str, int] = defaultdict(int)
        self.device = device
        self.warmup_iters = 20

    @contextmanager
    def record(self, name: str, *, sync: bool = False) -> Iterator[None]:
        """Time the enclosed block and attribute it to ``name``.

        Args:
            name: Stage identifier.
            sync: Synchronise the accelerator before and after. Required for
                any stage that queues device work.
        """
        if sync:
            sync_device(self.device)
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            if sync:
                sync_device(self.device)
            elapsed = time.perf_counter_ns() - start
            # Discard warmup: datumaro lazy-load, allocator growth, cuDNN autotune.
            if self._warmup[name] < self.warmup_iters:
                self._warmup[name] += 1
            else:
                self._stages.setdefault(name, StageStats(name)).samples.append(elapsed)

    def reset(self) -> None:
        """Clear all recorded timings and warmup counters."""
        self._stages.clear()
        self._warmup.clear()

    def attribution(self) -> list[dict[str, float | str | int]]:
        """Return per-stage rows sorted by descending total time.

        The ``share`` column is the fraction of *accounted* time, not of wall
        clock. A share table that sums to 1.0 tells you nothing about coverage —
        compare ``accounted_ms`` against measured wall time for that.
        """
        accounted = sum(s.total_ms for s in self._stages.values()) or 1.0
        rows = [
            {
                "stage": s.name,
                "count": s.count,
                "total_ms": round(s.total_ms, 3),
                "mean_ms": round(s.mean_ms, 4),
                "median_ms": round(s.median_ms, 4),
                "share": round(s.total_ms / accounted, 4),
            }
            for s in self._stages.values()
        ]
        return sorted(rows, key=lambda r: r["total_ms"], reverse=True)

    @property
    def accounted_ms(self) -> float:
        """Total time attributed across all stages, in milliseconds."""
        return sum(s.total_ms for s in self._stages.values())
