# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the XPU memory-clearing plugin.

Uses ``unittest.mock.patch`` rather than requiring real XPU hardware, so
this suite is portable to any CI runner — matching
``tests/unit/backend/ultralytics/plugins/test_xpu_support.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from getitune.backend.huggingface.plugins.xpu import XPUMemoryCallback, clear_xpu_memory


class TestClearXpuMemory:
    def test_is_a_no_op_when_xpu_is_unavailable(self) -> None:
        with patch("getitune.backend.huggingface.plugins.xpu.is_xpu_available", return_value=False):
            clear_xpu_memory()  # must not raise, must not touch torch.xpu

    def test_empties_the_xpu_cache_when_available(self) -> None:
        fake_torch = MagicMock()
        with (
            patch("getitune.backend.huggingface.plugins.xpu.is_xpu_available", return_value=True),
            patch.dict("sys.modules", {"torch": fake_torch}),
        ):
            clear_xpu_memory()
        fake_torch.xpu.empty_cache.assert_called_once()


class TestXPUMemoryCallback:
    def test_on_epoch_end_clears_memory(self) -> None:
        callback = XPUMemoryCallback()
        with patch("getitune.backend.huggingface.plugins.xpu.clear_xpu_memory") as mock_clear:
            callback.on_epoch_end(args=None, state=None, control=None)
        mock_clear.assert_called_once()
