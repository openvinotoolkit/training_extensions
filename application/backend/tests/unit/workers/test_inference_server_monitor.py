# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

from app.workers import InferenceServerMonitorThread


def _monitor(server: MagicMock, ticks: int = 1) -> InferenceServerMonitorThread:
    """A monitor thread whose run loop stops after the given number of ticks."""
    stop_event = MagicMock()
    stop_event.is_set.side_effect = [False] * ticks + [True]
    return InferenceServerMonitorThread(server=server, stop_event=stop_event)


class TestInferenceServerMonitorThread:
    def test_run_loop_evicts_expired_models(self) -> None:
        """Every tick asks the inference server to evict the models whose TTL has expired."""
        mock_server = MagicMock()
        monitor_thread = _monitor(mock_server, ticks=2)

        monitor_thread.run_loop()

        assert mock_server.evict_expired.call_count == 2
        assert monitor_thread._unload_failures == 0
        assert monitor_thread._next_unload_attempt == 0.0

    def test_run_loop_survives_evict_error(self) -> None:
        """A failed eviction must not kill the monitor thread, and schedules a retry after a backoff."""
        mock_server = MagicMock()
        mock_server.evict_expired.side_effect = RuntimeError("unload failed")
        monitor_thread = _monitor(mock_server)

        with patch("time.perf_counter", return_value=100):
            monitor_thread.run_loop()  # must not raise despite the eviction failure

        mock_server.evict_expired.assert_called_once_with()
        assert monitor_thread._unload_failures == 1
        assert monitor_thread._next_unload_attempt == 102  # first backoff: 2 ** 1 seconds

    def test_run_loop_skips_eviction_within_backoff(self) -> None:
        """While inside the backoff window, no eviction is attempted."""
        mock_server = MagicMock()
        monitor_thread = _monitor(mock_server)
        monitor_thread._unload_failures = 1
        monitor_thread._next_unload_attempt = 200.0

        with patch("time.perf_counter", return_value=100):
            monitor_thread.run_loop()

        mock_server.evict_expired.assert_not_called()
        assert monitor_thread._unload_failures == 1
        assert monitor_thread._next_unload_attempt == 200.0

    def test_run_loop_clears_retry_state_after_successful_eviction(self) -> None:
        """Once the backoff has elapsed, a successful eviction clears the retry bookkeeping."""
        mock_server = MagicMock()
        monitor_thread = _monitor(mock_server)
        monitor_thread._unload_failures = 3
        monitor_thread._next_unload_attempt = 150.0

        with patch("time.perf_counter", return_value=200):
            monitor_thread.run_loop()

        mock_server.evict_expired.assert_called_once_with()
        assert monitor_thread._unload_failures == 0
        assert monitor_thread._next_unload_attempt == 0.0
