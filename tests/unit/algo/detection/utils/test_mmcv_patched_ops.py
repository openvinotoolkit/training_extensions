# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of mmcv_patched_ops."""

import pytest
import torch
from mmcv.ops import nms
from otx.algo.detection.utils.mmcv_patched_ops import monkey_patched_nms
from mmcv.ops.roi_align import RoIAlign
from otx.algo.detection.utils import monkey_patched_roi_align


class TestMonkeyPatchedNMS:
    @pytest.fixture
    def setup(self):
        self.ctx = None
        self.bboxes = torch.tensor([[0.324, 0.422, 0.469, 0.123], [0.324, 0.422, 0.469, 0.123], [0.314, 0.423, 0.469, 0.123]])
        self.scores = torch.tensor([0.9, 0.2, 0.3])
        self.iou_threshold = 0.5
        self.offset = 0
        self.score_threshold = 0
        self.max_num = 0

    def test_case1(self, setup):
        # Testing when is_filtering_by_score is False
        result = monkey_patched_nms(self.ctx, self.bboxes, self.scores, self.iou_threshold, self.offset, self.score_threshold, self.max_num)
        assert torch.equal(result, torch.tensor([0, 2, 1]))

    def test_case2(self, setup):
        # Testing when is_filtering_by_score is True
        self.score_threshold = 0.8
        result = monkey_patched_nms(self.ctx, self.bboxes, self.scores, self.iou_threshold, self.offset, self.score_threshold, self.max_num)
        assert torch.equal(result, torch.tensor([0]))

    def test_case3(self, setup):
        # Testing when bboxes and scores have torch.bfloat16 dtype
        self.bboxes = torch.tensor([[0.324, 0.422, 0.469, 0.123], [0.324, 0.422, 0.469, 0.123], [0.314, 0.423, 0.469, 0.123]], dtype=torch.bfloat16)
        self.scores = torch.tensor([0.9, 0.2, 0.3], dtype=torch.bfloat16)
        result1 = monkey_patched_nms(self.ctx, self.bboxes, self.scores, self.iou_threshold, self.offset, self.score_threshold, self.max_num)
        assert torch.equal(result1, torch.tensor([0, 2, 1]))

    def test_case4(self, setup):
        # Testing when offset is not 0
        self.offset = 1
        result = monkey_patched_nms(self.ctx, self.bboxes, self.scores, self.iou_threshold, self.offset, self.score_threshold, self.max_num)
        assert torch.equal(result, torch.tensor([0]))

    def test_case5(self, setup):
        # Testing when max_num is greater than 0
        self.max_num = 1
        result = monkey_patched_nms(self.ctx, self.bboxes, self.scores, self.iou_threshold, self.offset, self.score_threshold, self.max_num)
        assert torch.equal(result, torch.tensor([0]))

    def test_case6(self, setup):
        # Testing that monkey_patched_nms equals mmcv nms
        self.score_threshold = 0.7
        result1 = monkey_patched_nms(self.ctx, self.bboxes, self.scores, self.iou_threshold, self.offset, self.score_threshold, self.max_num)
        result2 = nms(self.bboxes, self.scores, self.iou_threshold, score_threshold=self.score_threshold)
        assert torch.equal(result1, result2[1])
        # test random bboxes and scores
        bboxes = torch.rand((100, 4))
        scores = torch.rand(100)
        result1 = monkey_patched_nms(self.ctx, bboxes, scores, self.iou_threshold, self.offset, self.score_threshold, self.max_num)
        result2 = nms(bboxes, scores, self.iou_threshold, score_threshold=self.score_threshold)
        assert torch.equal(result1, result2[1])
        # no score threshold
        self.iou_threshold = 0.7
        self.score_threshold = 0.0
        result1 = monkey_patched_nms(self.ctx, bboxes, scores, self.iou_threshold, self.offset, self.score_threshold, self.max_num)
        result2 = nms(bboxes, scores, self.iou_threshold)
        assert torch.equal(result1, result2[1])


class TestMonkeyPatchedRoIAlign:
    @pytest.fixture
    def roi_align(self):
        return RoIAlign(output_size=(7, 7), spatial_scale=1.0, sampling_ratio=0, aligned=False, use_torchvision=True)

    @pytest.fixture
    def input(self):
        return torch.randn(1, 3, 32, 32)

    @pytest.fixture
    def rois(self):
        return torch.tensor([[0, 1, 10, 40, 50], [1, 2, 30, 70, 90]], dtype=torch.float)

    def test_monkey_patched_roi_align(self, roi_align, input, rois):
        expected_output = roi_align.forward(input, rois)
        output = monkey_patched_roi_align(roi_align, input, rois)
        assert torch.allclose(expected_output, output)
