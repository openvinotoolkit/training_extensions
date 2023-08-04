# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.v2.adapters.openvino.ops.generation import RangeV4


class TestRangeV4:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)


    def test_forward(self) -> None:
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
