# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMPretrain data transform functions."""
from __future__ import annotations

import pytest
from otx.core.data.entity.segmentation import SegDataEntity
from otx.core.data.transform_libs.mmcv import LoadImageFromFile
from otx.core.data.transform_libs.mmseg import LoadAnnotations, PackSegInputs


class TestLoadAnnotations:
    def test_transform(self, fxt_seg_data_entity) -> None:
        seg_data_entity: SegDataEntity = fxt_seg_data_entity[0]
        with pytest.raises(RuntimeError):
            new_results = LoadAnnotations().transform({})
        results = {"__otx__": seg_data_entity}
        new_results = LoadAnnotations().transform(results)
        assert isinstance(new_results, dict)
        assert "seg_fields" in new_results
        assert "gt_seg_map" in new_results["seg_fields"]
        assert new_results["gt_seg_map"].shape == seg_data_entity.img_info.img_shape


class TestPackSegInputs:
    def test_transform(self, fxt_seg_data_entity) -> None:
        instance: SegDataEntity = fxt_seg_data_entity[0]

        transforms = [
            LoadImageFromFile(),
            LoadAnnotations(),
            PackSegInputs(),
        ]

        for transform in transforms:
            instance = transform.transform(instance)

        assert isinstance(instance, SegDataEntity)
