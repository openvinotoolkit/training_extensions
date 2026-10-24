# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc

import pytest
import torch

from getitune.config import register_configs
from getitune.utils.device import is_xpu_available


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Cleanup memory after each test to prevent OOM errors in CI.

    This fixture only applies to tests under tests/unit/backend/lightning/models.
    """
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_xpu_available():
        torch.xpu.empty_cache()
