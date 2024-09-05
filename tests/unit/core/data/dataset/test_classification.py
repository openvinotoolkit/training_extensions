# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of classification datasets."""

from unittest.mock import MagicMock

from otx.core.data.dataset.classification import OTXHlabelClsDataset, OTXMulticlassClsDataset
from otx.core.data.entity.classification import MulticlassClsDataEntity


class TestOTXMulticlassClsDataset:
    def test_get_item(
        self,
        fxt_mock_dm_subset,
    ) -> None:
        dataset = OTXMulticlassClsDataset(
            dm_subset=fxt_mock_dm_subset,
            transforms=[lambda x: x],
            mem_cache_img_max_size=None,
            max_refetch=3,
        )
        assert isinstance(dataset[0], MulticlassClsDataEntity)

    def test_get_item_from_bbox_dataset(
        self,
        fxt_mock_det_dm_subset,
    ) -> None:
        dataset = OTXMulticlassClsDataset(
            dm_subset=fxt_mock_det_dm_subset,
            transforms=[lambda x: x],
            mem_cache_img_max_size=None,
            max_refetch=3,
        )
        assert isinstance(dataset[0], MulticlassClsDataEntity)


class TestOTXHlabelClsDataset:
    def test_add_ancestors(self, fxt_hlabel_dataset_subset):
        original_anns = fxt_hlabel_dataset_subset.get(id=0, subset="train").annotations
        assert len(original_anns) == 1

        hlabel_dataset = OTXHlabelClsDataset(
            dm_subset=fxt_hlabel_dataset_subset,
            transforms=MagicMock(),
        )
        # Added the ancestor
        adjusted_anns = hlabel_dataset.dm_subset.get(id=0, subset="train").annotations
        assert len(adjusted_anns) == 2
