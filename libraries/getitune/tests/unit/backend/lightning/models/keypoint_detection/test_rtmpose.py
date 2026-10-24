# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of RTMPose."""

import pytest
import torch
from torchvision import tv_tensors

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.keypoint_detection.rtmpose import RTMPose
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import SampleBatch


class TestRTMPoseTiny:
    @pytest.fixture
    def fxt_keypoint_det_model(self) -> RTMPose:
        return RTMPose(
            label_info=10,
            model_name="rtmpose_tiny",
            data_input_params=DataInputParams((192, 256), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    def test_customize_inputs(self, fxt_keypoint_det_model, fxt_keypoint_det_batch_data_entity):
        outputs = fxt_keypoint_det_model._customize_inputs(fxt_keypoint_det_batch_data_entity)
        entity = outputs["entity"]
        assert isinstance(entity.bboxes, list)
        assert isinstance(entity.bboxes[0], tv_tensors.BoundingBoxes)
        assert isinstance(entity.keypoints, list)
        assert isinstance(entity.keypoints[0], torch.Tensor)

    def test_customize_outputs(self, fxt_keypoint_det_model, fxt_keypoint_det_batch_data_entity):
        outputs = {"loss": torch.tensor(1.0)}
        fxt_keypoint_det_model.training = True
        preds = fxt_keypoint_det_model._customize_outputs(outputs, fxt_keypoint_det_batch_data_entity)
        assert isinstance(preds, BatchLoss)

        outputs = [(torch.randn(17, 2), torch.randn(17))]
        fxt_keypoint_det_model.training = False
        preds = fxt_keypoint_det_model._customize_outputs(outputs, fxt_keypoint_det_batch_data_entity)
        assert isinstance(preds, SampleBatch)
