# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pytest
from otx.core.data.mem_cache import MemCacheHandlerSingleton


@pytest.fixture(autouse=True)
def fxt_disable_mem_cache():
    """Disable mem cache to reduce memory usage."""

    original_mem_limit = MemCacheHandlerSingleton.CPU_MEM_LIMITS_GIB
    MemCacheHandlerSingleton.CPU_MEM_LIMITS_GIB = 99999999
    yield
    MemCacheHandlerSingleton.CPU_MEM_LIMITS_GIB = original_mem_limit
