# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import mmcv
import numpy as np
import pytest
import torch
from mmaction.registry import TRANSFORMS
from mmengine.fileio.file_client import FileClient
from otx.core.config.data import SubsetConfig
from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.action_detection import ActionDetDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.data.transform_libs.mmaction import (
    LoadAnnotations,
    LoadVideoForClassification,
    LoadVideoForDetection,
    MMActionTransformLib,
    PackActionInputs,
    RawFrameDecode,
)
from otx.core.types.transformer_libs import TransformLibType


class MockVideo:
    path: str = "video_path"


class TestActionClsPipeline:
    @pytest.fixture()
    def fxt_action_cls_data(self) -> dict:
        entity = ActionClsDataEntity(
            video=MockVideo(),
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


class TestActionDetPipelines:
    @pytest.fixture()
    def fxt_action_det_data(self, mocker) -> dict:
        entity = ActionDetDataEntity(
            image=torch.randn([3, 10, 10]),
            img_info=ImageInfo(
                img_idx=0,
                img_shape=(10, 10),
                ori_shape=(10, 10),
                image_color_channel=None,
            ),
            bboxes=torch.Tensor([[0, 0, 1, 1]]),
            labels=torch.LongTensor([[0, 1]]),
            frame_path="frame_dir/frame_path_001.png",
            proposals=np.array([0, 0, 1, 1]),
        )
        transform = LoadVideoForDetection()
        mocker.patch("os.listdir", return_value=range(10))
        return transform(entity)

    def test_load_video_for_detection(self, fxt_action_det_data):
        assert fxt_action_det_data["modality"] == "RGB"
        assert fxt_action_det_data["fps"] == 30
        assert fxt_action_det_data["timestamp_start"] == 900
        assert fxt_action_det_data["filename_tmpl"] == "{video}_{idx:04d}.{ext}"
        assert fxt_action_det_data["ori_shape"] == (10, 10)
        assert fxt_action_det_data["timestamp"] == 1
        assert fxt_action_det_data["shot_info"] == (1, 10)
        assert fxt_action_det_data["extension"] == "png"

    def test_load_annotations(self, fxt_action_det_data):
        transform = LoadAnnotations()
        out = transform(fxt_action_det_data)
        assert np.all(out["gt_bboxes"] == np.array([0, 0, 1, 1]))
        assert np.all(out["gt_labels"] == np.array([0, 1]))
        assert np.all(out["proposals"] == np.array([0, 0, 1, 1]))

    def test_raw_frame_decode(self, fxt_action_det_data, mocker):
        transform1 = LoadAnnotations()
        out = transform1(fxt_action_det_data)
        out["frame_inds"] = np.array([2])
        transform2 = RawFrameDecode()
        mocker.patch.object(FileClient, "get", return_value=None)
        mocker.patch.object(mmcv, "imfrombytes", return_value=np.ndarray([10, 10, 3]))
        out = transform2(out)
        assert out["original_shape"] == (10, 10)

    def test_pack_action_inputs(self, fxt_action_det_data, mocker):
        transform1 = LoadAnnotations()
        out = transform1(fxt_action_det_data)
        out["imgs"] = [np.ndarray([3, 10, 10])]
        out["original_shape"] = (10, 10)
        out["img_shape"] = (10, 10)
        transform2 = PackActionInputs()
        out = transform2(out)
        assert isinstance(out, ActionDetDataEntity)


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
            transforms=[{"type": "LoadVideoForDetection"}, {"type": "PackActionInputs"}],
            num_workers=2,
        )

        mocker.patch("otx.core.data.transform_libs.mmcv.convert_conf_to_mmconfig_dict", side_effect=mock_convert_func)
        transforms = MMActionTransformLib.generate(config)
        assert len(transforms) == 2
