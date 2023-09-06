"""Unit test for cusom max iou assigner."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.detection.adapters.mmdet.models.assigners import CustomMaxIoUAssigner
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestCustomMaxIoUAssigner:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Initial setup for unit tests."""
        self.assigner = CustomMaxIoUAssigner(
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            match_low_quality=True,
            ignore_iof_thr=-1,
            gpu_assign_thr=300,
        )
        self.assigner.cpu_assign_thr = 400

    @e2e_pytest_unit
    def test_assign_gpu(self):
        """Test custom assign function on gpu."""
        gt_bboxes = torch.randn(200, 4)
        bboxes = torch.randn(20000, 4)
        assign_result = self.assigner.assign(bboxes, gt_bboxes)
        assert assign_result.gt_inds.shape == torch.Size([20000])
        assert assign_result.max_overlaps.shape == torch.Size([20000])

    @e2e_pytest_unit
    def test_assign_cpu(self):
        """Test custom assign function on cpu."""
        gt_bboxes = torch.randn(350, 4)
        bboxes = torch.randn(20000, 4)
        assign_result = self.assigner.assign(bboxes, gt_bboxes)
        assert assign_result.gt_inds.shape == torch.Size([20000])
        assert assign_result.max_overlaps.shape == torch.Size([20000])

    @e2e_pytest_unit
    def test_assign_cpu_oom(self):
        """Test custom assign function on cpu in case of cpu oom."""
        gt_bboxes = torch.randn(450, 4)
        bboxes = torch.randn(20000, 4)
        assign_result = self.assigner.assign(bboxes, gt_bboxes)
        assert assign_result.gt_inds.shape == torch.Size([20000])
        assert assign_result.max_overlaps.shape == torch.Size([20000])

        self.assigner_cpu_assign_thr = 500
        new_assign_result = self.assigner.assign(bboxes, gt_bboxes)
        assert torch.all(new_assign_result.gt_inds == assign_result.gt_inds)
        assert torch.all(new_assign_result.max_overlaps == assign_result.max_overlaps)
