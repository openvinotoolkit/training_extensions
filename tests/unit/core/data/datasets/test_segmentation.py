# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import pytest
from datumaro import Dataset as DmDataset

from otx.core.data.dataset.segmentation import OTXSegmentationDataset
from otx.core.data.entity.segmentation import SegDataEntity

class TestOTXSegmentationDataset:
    @pytest.fixture()
    def dataset_generator(self) -> OTXSegmentationDataset:
        # load test dataset
        dm_dataset = DmDataset.import_from("tests/assets/common_semantic_segmentation_dataset/supervised",
                                           format="common_semantic_segmentation_with_subset_dirs").subsets()["train"]
        return OTXSegmentationDataset(dm_dataset, [])

    def test_get_item_impl(self, dataset_generator) -> None:
        item = dataset_generator._get_item_impl(2)
        assert item is not None
        assert isinstance(item, SegDataEntity)
        assert item.image.size != 0
        assert item.img_info.img_idx == 2
