"""Unit tests of Custom assigner for mmdet MaxIouAssigner."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from mmengine.config import Config

from otx.v2.adapters.torch.mmengine.mmdet.modules.models.assigners import CustomMaxIoUAssigner


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
        self.pred_instances = Config({})
        self.gt_instances = Config({})

    def test_assign_gpu(self):
        """Test custom assign function on gpu."""
        gt_bboxes = torch.randn(200, 4)
        gt_labels = torch.zeros(200, dtype=torch.int64)
        priors = torch.randn(20000, 4)
        self.gt_instances.bboxes = gt_bboxes
        self.gt_instances.labels = gt_labels
        self.pred_instances.priors = priors
        assign_result = self.assigner.assign(self.pred_instances, self.gt_instances)
        assert assign_result.gt_inds.shape == torch.Size([20000])
        assert assign_result.max_overlaps.shape == torch.Size([20000])

    def test_assign_cpu(self):
        """Test custom assign function on cpu."""
        gt_bboxes = torch.randn(350, 4)
        gt_labels = torch.zeros(350, dtype=torch.int64)
        priors = torch.randn(20000, 4)
        self.gt_instances.bboxes = gt_bboxes
        self.gt_instances.labels = gt_labels
        self.pred_instances.priors = priors
        assign_result = self.assigner.assign(self.pred_instances, self.gt_instances)
        assert assign_result.gt_inds.shape == torch.Size([20000])
        assert assign_result.max_overlaps.shape == torch.Size([20000])

    def test_assign_cpu_oom(self):
        """Test custom assign function on cpu in case of cpu oom."""
        gt_bboxes = torch.randn(450, 4)
        gt_labels = torch.zeros(450, dtype=torch.int64)
        priors = torch.randn(20000, 4)
        self.gt_instances.bboxes = gt_bboxes
        self.gt_instances.labels = gt_labels
        self.pred_instances.priors = priors
        assign_result = self.assigner.assign(self.pred_instances, self.gt_instances)
        assert assign_result.gt_inds.shape == torch.Size([20000])
        assert assign_result.max_overlaps.shape == torch.Size([20000])

        self.assigner_cpu_assign_thr = 500
        new_assign_result = self.assigner.assign(self.pred_instances, self.gt_instances)
        assert torch.all(new_assign_result.gt_inds == assign_result.gt_inds)
        assert torch.all(new_assign_result.max_overlaps == assign_result.max_overlaps)
