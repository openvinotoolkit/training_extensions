# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import psutil
import pytest


@pytest.fixture(autouse=True)
def hello():
    yield

    mem_info = psutil.virtual_memory()
    total_g = mem_info.total / 1024**3
    available_g = mem_info.available / 1024**3
    print(f"===== memory usage: {available_g}/{total_g} =====")
