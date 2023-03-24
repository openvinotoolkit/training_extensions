# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.heads.conv_head import (
    ConvClsHead,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestConvClsHead:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = ConvClsHead(
            num_classes=10,
            in_channels=3,
        )

    @e2e_pytest_unit
    def test_simple_test(self):
        input = torch.randn(10, 3, 1, 1)
        outputs = self.model.simple_test((input,), softmax=True)
        assert isinstance(outputs, list)
        assert len(outputs) == 10
        for output in outputs:
            assert np.isclose(output.sum(), np.array(1.0))

        outputs = self.model.simple_test((input,), post_process=False)
        assert isinstance(outputs, torch.Tensor)

    @e2e_pytest_unit
    def test_forward_train(self):
        input = torch.randn(10, 3, 1, 1)
        gt_label = torch.ones(10, dtype=torch.int64)
        output = self.model.forward_train((input,), gt_label)
        assert "loss" in output
