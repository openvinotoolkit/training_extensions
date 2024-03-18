# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import string

import numpy as np
import psutil
import pytest
from otx.core.data.mem_cache import (
    MemCacheHandlerSingleton,
    parse_mem_cache_size_to_int,
)


@pytest.fixture()
def fxt_data_list() -> list:
    bg = np.random.MT19937(seed=3003)
    rg = np.random.Generator(bg)
    num_data = 10
    h = w = key_len = 16

    data_list = []
    for _ in range(num_data):
        data = rg.integers(0, 256, size=[h, w, 3], dtype=np.uint8)
        key = "".join(
            [
                string.ascii_lowercase[i]
                for i in rg.integers(
                    0,
                    len(string.ascii_lowercase),
                    size=[key_len],
                )
            ],
        )
        meta = {
            "key": key,
        }
        data_list += [(key, data, meta)]

    return data_list


def get_data_list_size(data_list: list) -> int:
    size = 0
    for _, data, _ in data_list:
        size += data.size
    return size


class TestMemCacheHandler:
    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_cpu_limits(self, mode) -> None:
        memory_info = psutil.virtual_memory()
        total_mem_size_in_giga_bytes = int(memory_info.total / (1024**3))
        mem_size = total_mem_size_in_giga_bytes - (MemCacheHandlerSingleton.CPU_MEM_LIMITS_GIB - 5)
        handler = MemCacheHandlerSingleton.create(mode, mem_size * (1024**3))
        assert handler.mem_size == 0

    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_fully_caching(self, mode, fxt_data_list, monkeypatch) -> None:
        mem_size = get_data_list_size(fxt_data_list)
        monkeypatch.setattr(MemCacheHandlerSingleton, "check_system_memory", lambda *_: True)
        handler = MemCacheHandlerSingleton.create(mode, mem_size)

        for key, data, meta in fxt_data_list:
            assert handler.put(key, data, meta) > 0

        for key, data, meta in fxt_data_list:
            get_data, get_meta = handler.get(key)

            assert np.array_equal(get_data, data)
            assert get_meta == meta

        # Fully cached
        assert len(handler) == len(fxt_data_list)

    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_unfully_caching(self, mode, fxt_data_list, monkeypatch) -> None:
        mem_size = get_data_list_size(fxt_data_list) // 2
        monkeypatch.setattr(MemCacheHandlerSingleton, "check_system_memory", lambda *_: True)
        handler = MemCacheHandlerSingleton.create(mode, mem_size)

        for idx, (key, data, meta) in enumerate(fxt_data_list):
            if idx < len(fxt_data_list) // 2:
                assert handler.put(key, data, meta) > 0
            else:
                assert handler.put(key, data, meta) is None

        for idx, (key, data, meta) in enumerate(fxt_data_list):
            get_data, get_meta = handler.get(key)

            if idx < len(fxt_data_list) // 2:
                assert np.array_equal(get_data, data)
                assert get_meta == meta
            else:
                assert get_data is None

        # Unfully (half) cached
        assert len(handler) == len(fxt_data_list) // 2


@pytest.mark.parametrize(
    ("mem_size_arg", "expected"),
    [
        ("1561", 1561),
        ("121k", 121 * (2**10)),
        ("121kb", 121 * (10**3)),
        ("121kib", 121 * (2**10)),
        ("121m", 121 * (2**20)),
        ("121mb", 121 * (10**6)),
        ("121mib", 121 * (2**20)),
        ("121g", 121 * (2**30)),
        ("121gb", 121 * (10**9)),
        ("121gib", 121 * (2**30)),
        ("121as", None),
        ("121dddd", None),
    ],
)
def test_parse_mem_size_str(mem_size_arg, expected) -> None:
    try:
        assert parse_mem_cache_size_to_int(mem_size_arg) == expected
    except ValueError:
        assert expected is None
