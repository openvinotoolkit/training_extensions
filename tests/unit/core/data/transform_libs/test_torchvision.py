# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import numpy as np
import torch
from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.data.transform_libs.torchvision import DecodeVideo, PackVideo


class MockFrame:
    data = np.ndarray([3, 10, 10])


class MockVideo:
    data = [MockFrame()] * 10

    def __getitem__(self, idx):
        return self.data[idx]

    def close(self):
        return


class TestDecodeVideo:
    def test_train_case(self):
        transform = DecodeVideo(test_mode=False)
        video = MockVideo()
        assert len(transform._transform(video, {})) == 8

        transform = DecodeVideo(test_mode=False, out_of_bound_opt="repeat_last")
        assert len(transform._transform(video, {})) == 8

    def test_eval_case(self):
        transform = DecodeVideo(test_mode=True)
        video = MockVideo()
        assert len(transform._transform(video, {})) == 8

        transform = DecodeVideo(test_mode=True, out_of_bound_opt="repeat_last")
        assert len(transform._transform(video, {})) == 8


class TestPackVideo:
    def test_forward(self):
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
        transform = PackVideo()
        out = transform(entity)
        assert out.image == entity.video
