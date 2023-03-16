# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import string
from unittest.mock import patch

import numpy as np
import pytest
from torch.utils.data import DataLoader, Dataset

from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.image import Image
from otx.core.data.caching import MemCacheHandlerSingleton
from otx.core.data.pipelines.load_image_from_otx_dataset import LoadImageFromOTXDataset


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
def fxt_caching_dataset_cls(fxt_data_list: list):
    class CachingDataset(Dataset):
        def __init__(self) -> None:
            super().__init__()
            self.d_items = [
                DatasetItemEntity(
                    media=Image(data=data),
                    annotation_scene=AnnotationSceneEntity(annotations=[], kind=AnnotationSceneKind.ANNOTATION),
                )
                for _, data in fxt_data_list
            ]
            self.load = LoadImageFromOTXDataset()

        def __len__(self):
            return len(self.d_items)

        def __getitem__(self, index):
            d_item = self.d_items[index]

            results = {
                "dataset_item": d_item,
                "height": d_item.media.numpy.shape[0],
                "width": d_item.media.numpy.shape[1],
                "index": index,
            }

            results = self.load(results)
            return results["img"]

    yield CachingDataset


def get_data_list_size(data_list):
    size = 0
    for _, data in data_list:
        size += data.size
    return size


class TestMemCacheHandler:
    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_fully_caching(self, mode, fxt_data_list):
        mem_size = get_data_list_size(fxt_data_list)
        MemCacheHandlerSingleton.create(mode, mem_size)
        handler = MemCacheHandlerSingleton.get()

        for key, data in fxt_data_list:
            assert handler.put(key, data) > 0

        for key, data in fxt_data_list:
            get_data = handler.get(key)

            assert np.array_equal(get_data, data)

        # Fully cached
        assert len(handler) == len(fxt_data_list)

    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_unfully_caching(self, mode, fxt_data_list):
        mem_size = get_data_list_size(fxt_data_list) // 2
        MemCacheHandlerSingleton.create(mode, mem_size)
        handler = MemCacheHandlerSingleton.get()

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


class TestLoadImageFromFileWithCache:
    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_combine_with_dataloader(self, mode, fxt_caching_dataset_cls, fxt_data_list):
        mem_size = get_data_list_size(fxt_data_list)
        MemCacheHandlerSingleton.create(mode, mem_size)

        dataset = fxt_caching_dataset_cls()

        with patch(
            "otx.core.data.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data in fxt_data_list],
        ) as mock:
            for _ in DataLoader(dataset):
                continue

            # This initial round requires all data samples to be read from disk.
            assert mock.call_count == len(dataset)

        with patch(
            "otx.core.data.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data in fxt_data_list],
        ) as mock:
            for _ in DataLoader(dataset):
                continue

            # The second round requires no read.
            assert mock.call_count == 0
