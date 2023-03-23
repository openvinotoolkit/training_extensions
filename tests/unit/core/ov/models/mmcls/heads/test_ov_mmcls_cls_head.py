# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.heads.cls_head import ClsHead
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestClsHead:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = ClsHead(
            do_squeeze=True,
        )

    @e2e_pytest_unit
    def test_forward_train(self):
        cls_score = torch.randn(10, 100)
        gt_label = torch.ones(10, dtype=torch.int64)
        output = self.model.forward_train(cls_score, gt_label)
        assert "loss" in output

    @e2e_pytest_unit
    def test_simple_test(self):
        cls_score = torch.randn(10, 100)
        outputs = self.model.simple_test(cls_score)
        assert isinstance(outputs, list)
        assert len(outputs) == 10
