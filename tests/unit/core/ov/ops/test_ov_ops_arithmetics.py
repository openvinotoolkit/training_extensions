# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch

from otx.core.ov.ops.arithmetics import AddV1, DivideV1, MultiplyV1, SubtractV1, TanV0
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMultiplyV1:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 256)

    @e2e_pytest_unit
    def test_forward(self):
        op = MultiplyV1("dummy", shape=self.shape, auto_broadcast="none")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 * input_2)

        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape[1:])
        with pytest.raises(AssertionError):
            output = op(input_1, input_2)

        op = MultiplyV1("dummy", shape=self.shape, auto_broadcast="numpy")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 * input_2)

        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape[1:])
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 * input_2)

        op = MultiplyV1("dummy", shape=self.shape, auto_broadcast="dummy")
        with pytest.raises(NotImplementedError):
            output = op(input_1, input_2)


class TestDivideV1:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 256)

    @e2e_pytest_unit
    def test_forward(self):
        op = DivideV1("dummy", shape=self.shape, auto_broadcast="none")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 / input_2)

        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape[1:])
        with pytest.raises(AssertionError):
            output = op(input_1, input_2)

        op = DivideV1("dummy", shape=self.shape, auto_broadcast="numpy")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 / input_2)

        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape[1:])
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 / input_2)

        op = DivideV1("dummy", shape=self.shape, auto_broadcast="dummy")
        with pytest.raises(NotImplementedError):
            output = op(input_1, input_2)


class TestAddV1:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 256)

    @e2e_pytest_unit
    def test_forward(self):
        op = AddV1("dummy", shape=self.shape, auto_broadcast="none")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 + input_2)

        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape[1:])
        with pytest.raises(AssertionError):
            output = op(input_1, input_2)

        op = AddV1("dummy", shape=self.shape, auto_broadcast="numpy")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 + input_2)

        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape[1:])
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 + input_2)

        op = AddV1("dummy", shape=self.shape, auto_broadcast="dummy")
        with pytest.raises(NotImplementedError):
            output = op(input_1, input_2)


class TestSubtractV1:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 256)

    @e2e_pytest_unit
    def test_forward(self):
        op = SubtractV1("dummy", shape=self.shape, auto_broadcast="none")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 - input_2)

        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape[1:])
        with pytest.raises(AssertionError):
            output = op(input_1, input_2)

        op = SubtractV1("dummy", shape=self.shape, auto_broadcast="numpy")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 - input_2)

        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape[1:])
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 - input_2)

        op = SubtractV1("dummy", shape=self.shape, auto_broadcast="dummy")
        with pytest.raises(NotImplementedError):
            output = op(input_1, input_2)


class TestTanV0:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    @e2e_pytest_unit
    def test_forward(self):
        op = TanV0("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, torch.tan(self.input))
