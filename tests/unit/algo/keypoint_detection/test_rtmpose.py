# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of RTMPose."""

import pytest
import torch
from otx.algo.keypoint_detection.rtmpose import RTMPoseTiny
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity
from torchvision import tv_tensors


class TestRTMPoseTiny:
    @pytest.fixture()
    def fxt_keypoint_det_model(self) -> RTMPoseTiny:
        return RTMPoseTiny(label_info=10)

    def test_customize_inputs(self, fxt_keypoint_det_model, fxt_keypoint_det_batch_data_entity):
        outputs = fxt_keypoint_det_model._customize_inputs(fxt_keypoint_det_batch_data_entity)
        entity = outputs["entity"]
        assert isinstance(entity.bboxes, tv_tensors.BoundingBoxes)
        assert isinstance(entity.keypoints, torch.Tensor)
        assert isinstance(entity.keypoints_visible, torch.Tensor)

    def test_customize_outputs(self, fxt_keypoint_det_model, fxt_keypoint_det_batch_data_entity):
        outputs = {"loss": torch.tensor(1.0)}
        fxt_keypoint_det_model.training = True
        preds = fxt_keypoint_det_model._customize_outputs(outputs, fxt_keypoint_det_batch_data_entity)
        assert isinstance(preds, OTXBatchLossEntity)

        outputs = [(torch.randn((2, 17, 2)), torch.randn((2, 17)))]
        fxt_keypoint_det_model.training = False
        preds = fxt_keypoint_det_model._customize_outputs(outputs, fxt_keypoint_det_batch_data_entity)
        assert isinstance(preds, KeypointDetBatchDataEntity)
