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
from otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset import (
    LoadImageFromOTXDataset,
    LoadResizeDataFromOTXDataset,
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
        meta = {
            "key": key,
        }
        data_list += [(key, data, meta)]

    return data_list


@pytest.fixture
def fxt_caching_dataset_cls(fxt_data_list: list):
    class CachingDataset(Dataset):
        def __init__(self, enable_memcache: bool = True, load_resize: bool = False) -> None:
            super().__init__()
            self.d_items = [
                DatasetItemEntity(
                    media=Image(data=data),
                    annotation_scene=AnnotationSceneEntity(annotations=[], kind=AnnotationSceneKind.ANNOTATION),
                )
                for _, data, _ in fxt_data_list
            ]
            if load_resize == False:
                self.load = LoadImageFromOTXDataset(enable_memcache=enable_memcache)
            else:
                self.load = LoadResizeDataFromOTXDataset({}, enable_memcache=enable_memcache)

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
    for _, data, _ in data_list:
        size += data.size
    return size


class TestLoadImageFromFileWithCache:
    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_combine_with_dataloader(self, mode, fxt_caching_dataset_cls, fxt_data_list):
        mem_size = get_data_list_size(fxt_data_list)
        MemCacheHandlerSingleton.create(mode, mem_size)

        dataset = fxt_caching_dataset_cls()

        with patch(
            "otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data, _ in fxt_data_list],
        ) as mock:
            for _ in DataLoader(dataset):
                continue

            # This initial round requires all data samples to be read from disk.
            assert mock.call_count == len(dataset)

        with patch(
            "otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data, _ in fxt_data_list],
        ) as mock:
            for _ in DataLoader(dataset):
                continue

            # The second round requires no read.
            assert mock.call_count == 0

    @pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
    def test_disable_mem_cache(self, mode, fxt_caching_dataset_cls, fxt_data_list):
        mem_size = get_data_list_size(fxt_data_list)
        MemCacheHandlerSingleton.create(mode, mem_size)

        dataset = fxt_caching_dataset_cls(enable_memcache=False)

        with patch(
            "otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data, _ in fxt_data_list],
        ) as mock:
            for _ in DataLoader(dataset):
                continue

            # This initial round requires all data samples to be read from disk.
            assert mock.call_count == len(dataset)

        with patch(
            "otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data, _ in fxt_data_list],
        ) as mock:
            for _ in DataLoader(dataset):
                continue

            # The second round goes the same due to no cache support
            assert mock.call_count == len(dataset)


class TestLoadResizeDataFromOTXDataset:
    def test_init_assertion_error(self):
        with patch.object(LoadResizeDataFromOTXDataset, "_create_resize_op", return_value={}):
            with pytest.raises(AssertionError):
                op = LoadResizeDataFromOTXDataset(
                    load_ann_cfg=None,
                    resize_cfg={"size": [(1, 1), (2, 2)]},
                )

    def test_disable_memcache(self, fxt_caching_dataset_cls, fxt_data_list):
        dataset = fxt_caching_dataset_cls(enable_memcache=False, load_resize=True)

        with patch(
            "otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data, _ in fxt_data_list],
        ):
            with patch.object(dataset.load, "_get_unique_key") as mock:
                for _ in DataLoader(dataset):
                    continue
                assert mock.call_count == 0

    def test_enable_memcache(self, fxt_caching_dataset_cls, fxt_data_list):
        mem_size = get_data_list_size(fxt_data_list)
        MemCacheHandlerSingleton.create("singleprocessing", mem_size)

        dataset = fxt_caching_dataset_cls(load_resize=True)

        with patch(
            "otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data, _ in fxt_data_list],
        ) as mock:
            for _ in DataLoader(dataset):
                continue

            # This initial round requires all data samples to be read from disk.
            assert mock.call_count == len(dataset)

        with patch(
            "otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset.get_image",
            side_effect=[data for _, data, _ in fxt_data_list],
        ) as mock:
            for _ in DataLoader(dataset):
                continue

            # The second round requires no read.
            assert mock.call_count == 0


@pytest.mark.parametrize("mode", ["singleprocessing", "multiprocessing"])
def test_memcache_image_itemtype(mode):
    img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    MemCacheHandlerSingleton.create(mode, img.size * img.itemsize)
    cache = MemCacheHandlerSingleton.get()
    cache.put("img_u8", img)
    img_cached, _ = cache.get("img_u8")
    assert np.array_equal(img, img_cached)
    img = np.random.rand(10, 10, 3).astype(np.float)
    MemCacheHandlerSingleton.create(mode, img.size * img.itemsize)
    cache = MemCacheHandlerSingleton.get()
    cache.put("img_f32", img)
    img_cached, _ = cache.get("img_f32")
    assert np.array_equal(img, img_cached)
