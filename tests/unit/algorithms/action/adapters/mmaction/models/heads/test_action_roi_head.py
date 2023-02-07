"""Unit Test for otx.algorithms.action.adapters.mmaction.heads.roi_head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import torch
from mmcv.utils import Config

from otx.algorithms.action.adapters.mmaction.models.heads.roi_head import AVARoIHead
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockAVARoIHead(AVARoIHead):
    """Mock class for AVARoIHead."""

    def __init__(self):
        self.bbox_head = Config()
        self.test_cfg = Config()
        self.bbox_head.num_classes = 3
        self.test_cfg.action_thr = 0.5


class TestAVARoIHead:
    """Check AVARoIHead class.

    1. Check simple_test function
    <Steps>
        1. Generate sample tensor(1, 432, 32, 8, 11)
        2. Generate proposal_list: List[Tensor]
        3. Generate img_metas: List[Dict[str, Any]]
        4. Check bbox_results
            4-1. Check output lenth is fit with num_classes
            4-2. Check output has appropriate information(x1, y1, x2, y2, conf)
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.roi_head = AVARoIHead(
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor3D", roi_layer_type="RoIAlign", output_size=8, with_temporal_pool=True
            ),
            bbox_head=dict(type="BBoxHeadAVA", in_channels=432, num_classes=81, multilabel=False, dropout_ratio=0.5),
        )
        self.roi_head.test_cfg = Config()
        self.roi_head.test_cfg.action_thr = 0.5

    @e2e_pytest_unit
    def test_simple_test(self, mocker) -> None:
        """Test simple test function."""

        sample_input = torch.randn(1, 432, 32, 8, 1)
        proposal_list = [torch.Tensor([[0, 0, 10, 10]])]
        img_metas = [{"scores": np.array([1.0]), "img_shape": (256, 256)}]

        with torch.no_grad():
            out = self.roi_head.simple_test(sample_input, proposal_list, img_metas)

        assert len(out[0]) == self.roi_head.bbox_head.num_classes - 1
        assert out[0][0].shape[1] == 5

        mocker.patch(
            "otx.algorithms.action.adapters.mmaction.models.heads.roi_head.is_in_onnx_export", return_value=True
        )
        with torch.no_grad():
            bboxes, labels = self.roi_head.simple_test(sample_input, proposal_list, img_metas)
            assert isinstance(bboxes, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert bboxes.shape[-1] == 4
            assert labels.shape[-1] == self.roi_head.bbox_head.num_classes
