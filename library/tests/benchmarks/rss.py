# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Peak-RSS sampling across DataLoader worker processes."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

import psutil


class PeakRSSSampler:
    """Poll RSS of the current process and its children on a background thread.

    Polling rather than a single end-of-run reading: worker RSS peaks while a
    batch of dense ``(N, H, W)`` masks is alive and drops again after collate,
    so a post-hoc measurement misses it entirely.
    """

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_total_mb = 0.0
        self.peak_worker_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                total = proc.memory_info().rss
                worker_peak = 0
                for child in proc.children(recursive=True):
                    try:
                        rss = child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    total += rss
                    worker_peak = max(worker_peak, rss)
                self.peak_total_mb = max(self.peak_total_mb, total / 2**20)
                self.peak_worker_mb = max(self.peak_worker_mb, worker_peak / 2**20)
            except psutil.Error:
                pass
            self._stop.wait(self.interval_s)

    @contextmanager
    def sampling(self) -> Iterator[PeakRSSSampler]:
        """Sample RSS for the duration of the enclosed block."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        try:
            yield self
        finally:
            self._stop.set()
            self._thread.join(timeout=2.0)
