# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch

from otx.core.ov.ops.matmuls import EinsumV7, MatMulV0
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMatMulV0:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (10, 3, 5)
        self.input_1 = torch.randn(10, 3, 4)
        self.input_2 = torch.randn(10, 4, 5)

    @e2e_pytest_unit
    def test_forward(self):
        op = MatMulV0("dummy", shape=(self.shape,), transpose_a=False, transpose_b=False)
        output = op(self.input_1, self.input_2)
        assert output.shape == self.shape
        assert torch.equal(output, torch.matmul(self.input_1, self.input_2))

        op = MatMulV0("dummy", shape=(self.shape,), transpose_a=True, transpose_b=False)
        output = op(self.input_1.permute(0, 2, 1), self.input_2)
        assert output.shape == self.shape
        assert torch.equal(output, torch.matmul(self.input_1, self.input_2))

        op = MatMulV0("dummy", shape=(self.shape,), transpose_a=False, transpose_b=True)
        output = op(self.input_1, self.input_2.permute(0, 2, 1))
        assert output.shape == self.shape
        assert torch.equal(output, torch.matmul(self.input_1, self.input_2))

        op = MatMulV0("dummy", shape=(self.shape,), transpose_a=True, transpose_b=True)
        output = op(self.input_1.permute(0, 2, 1), self.input_2.permute(0, 2, 1))
        assert output.shape == self.shape
        assert torch.equal(output, torch.matmul(self.input_1, self.input_2))


class TestEinsumV7:
    @e2e_pytest_unit
    def test_forward(self):
        input = torch.randn(4, 4)
        op = EinsumV7("dummy", shape=(1,), equation="ii")
        output = op(input)
        assert torch.equal(output, torch.einsum("ii", input))

        input_1 = torch.randn(3, 2, 5)
        input_2 = torch.randn(3, 5, 4)
        op = EinsumV7("dummy", shape=(1,), equation="bij,bjk->bik")
        output = op(input_1, input_2)
        assert torch.equal(output, torch.einsum("bij,bjk->bik", input_1, input_2))
