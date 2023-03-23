# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch

from otx.core.ov.ops.generation import RangeV4
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestRangeV4:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)

    @e2e_pytest_unit
    def test_forward(self):
        op = RangeV4("dummy", shape=self.shape, output_type="i32")
        assert torch.equal(
            op(torch.tensor(0), torch.tensor(10), torch.tensor(1)),
            torch.arange(0, 10, 1),
        )

        op = RangeV4("dummy", shape=self.shape, output_type="f32")
        assert torch.equal(
            op(torch.tensor(0), torch.tensor(10), torch.tensor(1)),
            torch.arange(0, 10, 1, dtype=torch.float32),
        )
