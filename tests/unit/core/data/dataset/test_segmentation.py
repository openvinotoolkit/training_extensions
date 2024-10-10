# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests of classification datasets."""

from otx.core.data.dataset.segmentation import OTXSegmentationDataset
from otx.core.data.entity.segmentation import SegDataEntity


class TestOTXSegmentationDataset:
    def test_get_item(
        self,
        fxt_mock_dm_subset,
    ) -> None:
        dataset = OTXSegmentationDataset(
            dm_subset=fxt_mock_dm_subset,
            transforms=[lambda x: x],
            mem_cache_img_max_size=None,
            max_refetch=3,
        )
        assert isinstance(dataset[0], SegDataEntity)
        assert "otx_background_lbl" in [label_name.lower() for label_name in dataset.label_info.label_names]

    def test_get_item_from_bbox_dataset(
        self,
        fxt_mock_det_dm_subset,
    ) -> None:
        dataset = OTXSegmentationDataset(
            dm_subset=fxt_mock_det_dm_subset,
            transforms=[lambda x: x],
            mem_cache_img_max_size=None,
            max_refetch=3,
        )
        assert isinstance(dataset[0], SegDataEntity)
        # OTXSegmentationDataset should add background when getting a dataset which includes only bbox annotations
        assert "otx_background_lbl" in [label_name.lower() for label_name in dataset.label_info.label_names]
