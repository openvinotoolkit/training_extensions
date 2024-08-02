# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import HlabelClsDataEntity, MulticlassClsDataEntity, MultilabelClsDataEntity
from otx.core.types.transformer_libs import TransformLibType

SKIP_MMLAB_TEST = False
try:
    from mmpretrain.registry import TRANSFORMS
    from otx.core.data.transform_libs.mmcv import LoadImageFromFile
    from otx.core.data.transform_libs.mmpretrain import MMPretrainTransformLib, PackInputs
except ImportError:
    SKIP_MMLAB_TEST = True


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
class TestPackInputs:
    @pytest.mark.parametrize(
        "entity",
        [
            MulticlassClsDataEntity(
                image=np.ndarray([3, 10, 10]),
                img_info=ImageInfo(
                    img_idx=0,
                    img_shape=(10, 10),
                    ori_shape=(10, 10),
                ),
                labels=torch.LongTensor([0]),
            ),
            MultilabelClsDataEntity(
                image=np.ndarray([3, 10, 10]),
                img_info=ImageInfo(
                    img_idx=0,
                    img_shape=(10, 10),
                    ori_shape=(10, 10),
                ),
                labels=torch.LongTensor([0, 1]),
            ),
            HlabelClsDataEntity(
                image=np.ndarray([3, 10, 10]),
                img_info=ImageInfo(
                    img_idx=0,
                    img_shape=(10, 10),
                    ori_shape=(10, 10),
                ),
                labels=torch.as_tensor([0]),
            ),
        ],
    )
    def test_transform(self, entity):
        assert isinstance(PackInputs()(LoadImageFromFile()(entity)), type(entity))


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
class TestMMPretrainTransformLib:
    def test_get_builder(self) -> None:
        assert MMPretrainTransformLib.get_builder() == TRANSFORMS

    def test_generate(self, mocker) -> None:
        def mock_convert_func(cfg: dict) -> dict:
            return cfg

        config = SubsetConfig(
            batch_size=64,
            subset_name="train",
            transform_lib_type=TransformLibType.MMPRETRAIN,
            transforms=[{"type": "LoadImageFromFile"}, {"type": "PackInputs"}],
            num_workers=2,
        )

        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        transforms = MMPretrainTransformLib.generate(config)
        assert len(transforms) == 2
