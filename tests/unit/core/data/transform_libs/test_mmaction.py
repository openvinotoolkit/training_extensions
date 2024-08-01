# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.core.config.data import SubsetConfig
from otx.core.types.transformer_libs import TransformLibType

SKIP_MMLAB_TEST = False
try:
    from mmaction.registry import TRANSFORMS
    from otx.core.data.transform_libs.mmaction import MMActionTransformLib
except ImportError:
    SKIP_MMLAB_TEST = True


class MockVideo:
    path: str = "video_path"


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
class TestMMActionTransformLib:
    def test_get_builder(self) -> None:
        assert MMActionTransformLib.get_builder() == TRANSFORMS

    def test_generate(self, mocker) -> None:
        def mock_convert_func(cfg: dict) -> dict:
            return cfg

        config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMACTION,
            transforms=[{"type": "PackActionInputs"}],
            num_workers=2,
        )

        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        transforms = MMActionTransformLib.generate(config)
        assert len(transforms) == 1
