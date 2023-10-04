# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math

import pytest
import torch
from otx.v2.adapters.openvino.ops.activations import (
    ClampV0,
    EluV0,
    ExpV0,
    GeluV7,
    HardSigmoidV0,
    HSigmoidV5,
    HSwishV4,
    MishV4,
    PReluV0,
    ReluV0,
    SeluV0,
    SigmoidV0,
    SoftMaxV0,
    SoftMaxV1,
    SwishV4,
    TanhV0,
)
from torch.nn import functional


class TestSoftMaxV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = SoftMaxV0("dummy", shape=self.shape, axis=-1)
        output = op(self.input)
        assert torch.equal(output, functional.softmax(self.input, dim=op.attrs.axis))
        assert torch.equal(output >= 0, output <= 1)
        assert torch.allclose(output.sum(-1), torch.tensor(1.0))


class TestSoftMaxV1:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = SoftMaxV1("dummy", shape=self.shape, axis=-1)
        output = op(self.input)
        assert torch.equal(output, functional.softmax(self.input, dim=op.attrs.axis))
        assert torch.equal(output >= 0, output <= 1)
        assert torch.allclose(output.sum(-1), torch.tensor(1.0))


class TestReluV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = ReluV0("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, functional.relu(self.input))
        assert all((output >= 0).flatten())


class TestSwishV4:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = SwishV4("dummy", shape=self.shape)
        output = op(self.input, 1.0)
        assert torch.equal(output, self.input * torch.sigmoid(self.input * 1.0))


class TestSigmoidV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = SigmoidV0("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, torch.sigmoid(self.input))
        assert torch.equal(output >= 0, output <= 1)


class TestClampV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = ClampV0("dummy", shape=self.shape, min=-0.05, max=0.05)
        output = op(self.input)
        assert torch.equal(output, self.input.clamp(min=op.attrs.min, max=op.attrs.max))
        assert torch.equal(output >= op.attrs.min, output <= op.attrs.max)


class TestPReluV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = PReluV0("dummy", shape=self.shape)
        output = op(self.input, torch.tensor(0.1))
        assert torch.equal(output, functional.prelu(self.input, torch.tensor(0.1)))


class TestTanhV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = TanhV0("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, functional.tanh(self.input))


class TestEluV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = EluV0("dummy", shape=self.shape, alpha=0.1)
        output = op(self.input)
        assert torch.equal(output, functional.elu(self.input, alpha=op.attrs.alpha))


class TestSeluV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = SeluV0("dummy", shape=self.shape)
        output = op(self.input, 0.1, 0.1)
        assert torch.equal(output, 0.1 * functional.elu(self.input, alpha=0.1))


class TestMishV4:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = MishV4("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, self.input * functional.tanh(functional.softplus(self.input)))


class TestHSwishV4:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = HSwishV4("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, functional.hardswish(self.input))


class TestHSigmoidV5:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = HSigmoidV5("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, functional.hardsigmoid(self.input))


class TestExpV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = ExpV0("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, torch.exp(self.input))


class TestHardSigmoidV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        op = HardSigmoidV0("dummy", shape=self.shape)
        output = op(self.input, 0.1, 0.1)
        assert torch.equal(
            output,
            torch.maximum(
                torch.zeros_like(self.input),
                torch.minimum(torch.ones_like(self.input), self.input * 0.1 + 0.1),
            ),
        )


class TestGeluV7:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 256)
        self.input = torch.randn(self.shape)

    def test_forward(self) -> None:
        with pytest.raises(ValueError, match="Invalid approximation_mode"):
            GeluV7("dummy", shape=self.shape, approximation_mode="dummy")

        op = GeluV7("dummy", shape=self.shape)
        output = op(self.input)
        assert torch.equal(output, functional.gelu(self.input))

        op = GeluV7("dummy", shape=self.shape, approximation_mode="tanh")
        output = op(self.input)
        assert torch.equal(
            output,
            self.input
            * 0.5
            * (1 + functional.tanh(torch.sqrt(2 / torch.tensor(math.pi)) * (self.input + 0.044715 * self.input**3))),
        )
