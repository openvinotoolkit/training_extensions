# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import torch
from mmaction.registry import TRANSFORMS
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo, VideoInfo
from otx.core.data.transform_libs.mmaction import (
    LoadVideoForClassification,
    MMActionTransformLib,
    PackActionInputs,
)
from otx.core.types.transformer_libs import TransformLibType


class MockVideo:
    path: str = "video_path"


class TestActionClsPipeline:
    @pytest.fixture()
    def fxt_action_cls_data(self) -> dict:
        entity = ActionClsDataEntity(
            video=MockVideo(),
            video_info=VideoInfo(),
            image=[],
            img_info=ImageInfo(
                img_idx=0,
                img_shape=(0, 0),
                ori_shape=(0, 0),
                image_color_channel=None,
            ),
            labels=torch.LongTensor([0]),
        )
        transform = LoadVideoForClassification()
        return transform(entity)

    def test_load_video_for_classification(self, fxt_action_cls_data):
        assert fxt_action_cls_data["filename"] == "video_path"
        assert fxt_action_cls_data["start_index"] == 0
        assert fxt_action_cls_data["modality"] == "RGB"

    def test_pack_action_inputs(self, fxt_action_cls_data):
        transform = PackActionInputs()
        fxt_action_cls_data["imgs"] = [np.ndarray([3, 10, 10])]
        fxt_action_cls_data["original_shape"] = (10, 10)
        fxt_action_cls_data["img_shape"] = (10, 10)
        assert isinstance(transform(fxt_action_cls_data), ActionClsDataEntity)


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
        assert len(transforms) == 2
