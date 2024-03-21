# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import pytest
from mmseg.registry import TRANSFORMS
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.segmentation import SegDataEntity
from otx.core.data.transform_libs.mmcv import LoadImageFromFile
from otx.core.data.transform_libs.mmseg import LoadAnnotations, MMSegTransformLib, PackSegInputs
from otx.core.types.transformer_libs import TransformLibType


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


class TestMMSegTransformLib:
    def test_get_builder(self) -> None:
        assert MMSegTransformLib.get_builder() == TRANSFORMS

    def test_generate(self, mocker) -> None:
        def mock_convert_func(cfg: dict) -> dict:
            return cfg

        config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMSEG,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "LoadAnnotations"}, {"type": "PackSegInputs"}],
            num_workers=2,
        )

        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        transforms = MMSegTransformLib.generate(config)
        assert len(transforms) == 3
