# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os.path as osp
import string
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from otx.mpa.modules.datasets.pipelines.caching import (
    LoadImageFromFileWithCache,
    MemCacheHandler,
)


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
        data_list += [(key, data)]

    return data_list


@pytest.fixture
def fxt_caching_dataset_cls(fxt_data_list):
    with TemporaryDirectory() as img_prefix:
        for key, data in fxt_data_list:
            cv2.imwrite(osp.join(img_prefix, key + ".png"), data)

        class CachingDataset(Dataset):
            def __init__(self) -> None:
                super().__init__()
                self.data_list = fxt_data_list
                self.load = LoadImageFromFileWithCache()
                self.file_get_count = 0

                __get = self.load.file_client.get

                def _get(filepath):
                    self.file_get_count += 1
                    return __get(filepath)

                self.load.file_client.get = _get

            def reset_file_count(self):
                self.file_get_count = 0

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, index):
                key, _ = self.data_list[index]
                results = {
                    "img_prefix": img_prefix,
                    "img_info": {"filename": key + ".png"},
                }
                return self.load(results)

        yield CachingDataset


def get_data_list_size(data_list):
    size = 0
    for _, data in data_list:
        size += data.size
    return size


class TestMemCacheHandler:
    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_fully_caching(self, mode, fxt_data_list):
        mem_size = str(get_data_list_size(fxt_data_list))
        MemCacheHandler.create(mode, mem_size)
        handler = MemCacheHandler()

        for key, data in fxt_data_list:
            assert handler.put(key, data) > 0

        for key, data in fxt_data_list:
            get_data = handler.get(key)

            assert np.array_equal(get_data, data)

        # Fully cached
        assert len(handler) == len(fxt_data_list)

    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_unfully_caching(self, mode, fxt_data_list):
        mem_size = str(get_data_list_size(fxt_data_list) // 2)
        MemCacheHandler.create(mode, mem_size)
        handler = MemCacheHandler()

        for idx, (key, data) in enumerate(fxt_data_list):
            if idx < len(fxt_data_list) // 2:
                assert handler.put(key, data) > 0
            else:
                assert handler.put(key, data) is None

        for idx, (key, data) in enumerate(fxt_data_list):
            get_data = handler.get(key)

            if idx < len(fxt_data_list) // 2:
                assert np.array_equal(get_data, data)
            else:
                assert get_data is None

        # Unfully (half) cached
        assert len(handler) == len(fxt_data_list) // 2

    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    @pytest.mark.parametrize(
        "mem_size,expected",
        [
            ("1561", 1561),
            ("121k", 121 * (2**10)),
            ("121kb", 121 * (2**10)),
            ("121kib", 121 * (10**3)),
            ("121as", None),
            ("121dddd", None),
        ],
    )
    def test_mem_size_parsing(self, mode, mem_size, expected):
        try:
            MemCacheHandler.create(mode, mem_size)
            handler = MemCacheHandler()
            assert handler.mem_size == expected
        except ValueError:
            assert expected is None


class TestLoadImageFromFileWithCache:
    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_combine_with_dataloader(self, mode, fxt_caching_dataset_cls, fxt_data_list):
        mem_size = str(get_data_list_size(fxt_data_list))
        MemCacheHandler.create(mode, mem_size)

        dataset = fxt_caching_dataset_cls()

        for _ in DataLoader(dataset):
            continue

        # This initial round requires file_client.get() for all data samples.
        assert dataset.file_get_count == len(dataset)

        dataset.reset_file_count()

        for _ in DataLoader(dataset):
            continue

        # The second round requires no file_client.get().
        assert dataset.file_get_count == 0
