# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import pytest
import torch
from otx.v2.adapters.openvino.ops.arithmetics import AddV1, DivideV1, MultiplyV1, SubtractV1, TanV0


class TestMultiplyV1:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)


    def test_forward(self) -> None:
        op = MultiplyV1("dummy", shape=self.shape, auto_broadcast="none")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 * input_2)

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
    def setup(self) -> None:
        self.shape = (4, 256)


    def test_forward(self) -> None:
        op = DivideV1("dummy", shape=self.shape, auto_broadcast="none")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 / input_2)

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
    def setup(self) -> None:
        self.shape = (4, 256)


    def test_forward(self) -> None:
        op = AddV1("dummy", shape=self.shape, auto_broadcast="none")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 + input_2)

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
    def setup(self) -> None:
        self.shape = (4, 256)


    def test_forward(self) -> None:
        op = SubtractV1("dummy", shape=self.shape, auto_broadcast="none")
        input_1 = torch.randn(self.shape)
        input_2 = torch.randn(self.shape)
        output = op(input_1, input_2)
        assert torch.equal(output, input_1 - input_2)

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
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)


    def test_forward(self) -> None:
        op = TanV0("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, torch.tan(self.input))
