# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from multiprocessing.synchronize import Event

from loguru import logger

from app.services.inference import InferenceServer
from app.workers.base import BaseThreadWorker

# When evicting expired models fails, the attempt is retried with an exponential backoff
# (in seconds) capped at this value, instead of giving up. This prevents a single transient
# failure from leaving idle models loaded indefinitely, while avoiding hammering the server on
# every monitor tick.
MAX_UNLOAD_RETRY_BACKOFF_SECONDS = 30.0


class InferenceServerMonitorThread(BaseThreadWorker):
    """
    Inference Server Monitor Thread drives the time-based part of the inference server model cache.

    The inference server keeps a small cache of loaded models and tracks, per model, when it was last
    used. This thread periodically asks the server to evict the models whose TTL has expired, so that
    idle models do not keep occupying memory. Eviction failures are retried with an exponential
    backoff instead of being abandoned, and never terminate the monitoring loop.
    """

    ROLE = "InferenceServerMonitor"

    def __init__(self, server: InferenceServer, stop_event: Event) -> None:
        super().__init__(stop_event=stop_event)

        self._server = server

        # Retry bookkeeping for failed eviction attempts. While eviction fails, the next attempt is
        # delayed by an exponential backoff instead of running on every tick.
        self._unload_failures = 0
        self._next_unload_attempt = 0.0

    def setup(self) -> None:
        """No resources to allocate: the inference server is injected by the scheduler."""

    def run_loop(self) -> None:
        while not self.should_stop():
            now = time.perf_counter()
            if now >= self._next_unload_attempt:
                try:
                    self._server.evict_expired()
                    self._unload_failures = 0
                    self._next_unload_attempt = 0.0
                except Exception:
                    # A failure here must not kill the monitor thread; otherwise models would never
                    # be evicted again for the lifetime of the process. Schedule a backoff instead.
                    self._unload_failures += 1
                    backoff = min(2.0**self._unload_failures, MAX_UNLOAD_RETRY_BACKOFF_SECONDS)
                    self._next_unload_attempt = now + backoff
                    logger.exception(
                        "Failed to evict expired inference models (attempt {}); retrying in {}s.",
                        self._unload_failures,
                        backoff,
                    )

            self.stop_aware_sleep(1)
