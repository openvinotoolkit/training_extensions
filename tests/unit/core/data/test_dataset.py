# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from otx.core.data.dataset.action_classification import OTXActionClsDataset
from otx.core.data.dataset.classification import HLabelInfo


class TestDataset:
    def test_get_item(
        self,
        mocker,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset: MagicMock,
        fxt_mock_hlabelinfo,
    ) -> None:
        dataset_cls, data_entity_cls, kwargs = fxt_dataset_and_data_entity_cls
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=[lambda x: x],
            mem_cache_img_max_size=None,
            max_refetch=3,
            **kwargs,
        )
        dataset.num_classes = 1
        item = dataset[0]

        assert isinstance(item, data_entity_cls)
        fxt_mock_dm_subset.get.assert_called_once()

        mocker.patch.object(dataset, "_get_item_impl", return_value=None)
        with pytest.raises(RuntimeError):
            dataset[0]

    def test_sample_another_idx(
        self,
        mocker,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset,
        fxt_mock_hlabelinfo,
    ) -> None:
        dataset_cls, data_entity_cls, kwargs = fxt_dataset_and_data_entity_cls
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=lambda x: x,
            mem_cache_img_max_size=None,
            **kwargs,
        )
        dataset.num_classes = 1
        assert dataset._sample_another_idx() < len(dataset)

    @pytest.mark.parametrize("mem_cache_img_max_size", [(3, 5), (5, 3)])
    def test_mem_cache_resize(
        self,
        mocker,
        mem_cache_img_max_size,
        fxt_mem_cache_handler,
        fxt_dataset_and_data_entity_cls,
        fxt_mock_dm_subset: MagicMock,
        fxt_dm_item,
        fxt_mock_hlabelinfo,
    ) -> None:
        dataset_cls, data_entity_cls, kwargs = fxt_dataset_and_data_entity_cls
        mocker.patch.object(HLabelInfo, "from_dm_label_groups", return_value=fxt_mock_hlabelinfo)
        dataset = dataset_cls(
            dm_subset=fxt_mock_dm_subset,
            transforms=lambda x: x,
            mem_cache_handler=fxt_mem_cache_handler,
            mem_cache_img_max_size=mem_cache_img_max_size,
            **kwargs,
        )
        dataset.num_classes = 1

        item = dataset[0]  # Put in the cache

        # The returned image should be resized because it was resized before caching
        h_expected = w_expected = min(mem_cache_img_max_size)
        if dataset_cls != OTXActionClsDataset:  # Action classification dataset handle video, not image.
            assert item.image.shape[:2] == (h_expected, w_expected)
            assert item.img_info.img_shape == (h_expected, w_expected)

            item = dataset[0]  # Take from the cache

            assert item.image.shape[:2] == (h_expected, w_expected)
            assert item.img_info.img_shape == (h_expected, w_expected)
