"""Unit Test for otx.algorithms.action.adapters.mmaction.heads.movinet_head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.action.adapters.mmaction.models.heads.movinet_head import (
    MoViNetHead,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMoViNetHead:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.movinet_head = MoViNetHead(
            num_classes=400,
            in_channels=480,
            hidden_dim=2048,
            loss_cls=dict(type="CrossEntropyLoss", loss_weight=1.0),
        )

    @e2e_pytest_unit
    def test_forward(self) -> None:
        """Test forward function."""
        sample_input = torch.randn(1, 480, 1, 1, 1)
        with torch.no_grad():
            out = self.movinet_head(sample_input)
        assert out.shape == (1, self.movinet_head.num_classes)
