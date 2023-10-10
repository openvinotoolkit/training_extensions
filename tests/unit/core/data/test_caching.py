# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import string

import numpy as np
import pytest
import psutil

from otx.core.data.caching import MemCacheHandlerSingleton


@pytest.fixture
def fxt_data_list():
    np.random.seed(3003)

    num_data = 10
    h = w = key_len = 16

    data_list = []
    for _ in range(num_data):
        data = np.random.randint(0, 256, size=[h, w, 3], dtype=np.uint8)
        key = "".join(
            [string.ascii_lowercase[i] for i in np.random.randint(0, len(string.ascii_lowercase), size=[key_len])]
        )
        meta = {
            "key": key,
        }
        data_list += [(key, data, meta)]

    return data_list


def get_data_list_size(data_list):
    size = 0
    for _, data, _ in data_list:
        size += data.size
    return size


class TestMemCacheHandler:
    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_cpu_limits(self, mode):
        memory_info = psutil.virtual_memory()
        total_mem_size_GiB = int(memory_info.total / (1024**3))
        mem_size = total_mem_size_GiB - (MemCacheHandlerSingleton.CPU_MEM_LIMITS_GIB - 5)
        MemCacheHandlerSingleton.create(mode, mem_size * (1024**3))
        assert MemCacheHandlerSingleton.instance.mem_size == 0

    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_fully_caching(self, mode, fxt_data_list):
        mem_size = get_data_list_size(fxt_data_list)
        MemCacheHandlerSingleton.create(mode, mem_size)
        handler = MemCacheHandlerSingleton.get()

        for key, data, meta in fxt_data_list:
            assert handler.put(key, data, meta) > 0

        for key, data, meta in fxt_data_list:
            get_data, get_meta = handler.get(key)

            assert np.array_equal(get_data, data)
            assert get_meta == meta

        # Fully cached
        assert len(handler) == len(fxt_data_list)

    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_unfully_caching(self, mode, fxt_data_list):
        mem_size = get_data_list_size(fxt_data_list) // 2
        MemCacheHandlerSingleton.create(mode, mem_size)
        handler = MemCacheHandlerSingleton.get()

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
