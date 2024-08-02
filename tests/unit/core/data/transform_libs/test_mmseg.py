# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.segmentation import SegDataEntity
from otx.core.types.transformer_libs import TransformLibType

SKIP_MMLAB_TEST = False
try:
    from mmseg.registry import TRANSFORMS
    from otx.core.data.transform_libs.mmcv import LoadImageFromFile
    from otx.core.data.transform_libs.mmseg import LoadAnnotations, MMSegTransformLib, PackSegInputs
except ImportError:
    SKIP_MMLAB_TEST = True


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
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


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
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


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
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
