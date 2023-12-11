# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from datumaro.components.media import Image


class TestDataset:
    def test_get_item(
        self,
        mocker,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset: MagicMock,
    ) -> None:
        dataset_cls, data_entity_cls = fxt_dataset_and_data_entity_cls
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=lambda x: x,
            mem_cache_img_max_size=None,
        )
        item = dataset[0]

        assert isinstance(item, data_entity_cls)
        fxt_mock_dm_subset.get.assert_called_once()

        mocker.patch.object(dataset, "_get_item_impl", return_value=None)
        with pytest.raises(RuntimeError):
            dataset[0]

    def test_sample_another_idx(self, fxt_dataset_and_data_entity_cls, fxt_mock_dm_subset) -> None:
        dataset_cls, dataset_entity_cls = fxt_dataset_and_data_entity_cls
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=lambda x: x,
            mem_cache_img_max_size=None,
        )
        assert dataset._sample_another_idx() < len(dataset)

    @pytest.mark.parametrize("mem_cache_img_max_size", [(3, 5), (5, 3)])
    def test_mem_cache_resize(
        self,
        mem_cache_img_max_size,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset: MagicMock,
        fxt_dm_item,
    ) -> None:
        dataset_cls, data_entity_cls = fxt_dataset_and_data_entity_cls

        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=lambda x: x,
            mem_cache_img_max_size=mem_cache_img_max_size,
        )

        item = dataset[0]  # Put in the cache

        assert item.image.shape[:2] == fxt_dm_item.media_as(Image).size
        assert item.img_info.img_shape == fxt_dm_item.media_as(Image).size

        item = dataset[0]  # Take from the cache

        min_size = min(mem_cache_img_max_size)

        assert item.image.shape[:2] == (min_size, min_size)
        assert item.img_info.img_shape == (min_size, min_size)
