import psutil
import time

import pytest


@pytest.mark.skip(reason="only for the debugging")
def test_memory_bound2():
    print(psutil.virtual_memory())
    alloc = []
    alloc_unit = "0123456789" * 1024 * 1024 * 10  # 100m
    idx = 0
    while idx < 1024:
        alloc.append(f"{idx}{alloc_unit}")
        print(f"[{idx}]{psutil.virtual_memory()}")
        time.sleep(1)
        idx += 1
