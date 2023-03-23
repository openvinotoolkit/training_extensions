# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import torch

from otx.core.ov.ops.reductions import (
    ReduceMeanV1,
    ReduceMinV1,
    ReduceProdV1,
    ReduceSumV1,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestReduceMeanV1:
    @e2e_pytest_unit
    def test_forward(self):
        op = ReduceMeanV1("dummy", shape=(1,), keep_dims=False)
        input = torch.randn(6, 12, 10, 24)

        output = op(input, [])
        assert torch.equal(output, input)

        output = op(input, torch.tensor([1, 2]))
        assert output.shape == (6, 24)
        assert torch.equal(output, torch.mean(input, dim=(1, 2)))

        output = op(input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10)
        assert torch.equal(output, torch.mean(input, dim=-1))

        op = ReduceMeanV1("dummy", shape=(1,), keep_dims=True)

        output = op(input, [])
        assert torch.equal(output, input)

        output = op(input, torch.tensor([1, 2]))
        assert output.shape == (6, 1, 1, 24)
        assert torch.equal(output, torch.mean(input, dim=(1, 2), keepdim=True))

        output = op(input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10, 1)
        assert torch.equal(output, torch.mean(input, dim=-1, keepdim=True))


class TestReduceProdV1:
    @e2e_pytest_unit
    def test_forward(self):
        op = ReduceProdV1("dummy", shape=(1,), keep_dims=False)
        input = torch.randn(6, 12, 10, 24)

        output = op(input, [])
        assert torch.equal(output, input)

        output = op(input, torch.tensor([1, 2]))
        assert output.shape == (6, 24)

        output = op(input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10)

        op = ReduceProdV1("dummy", shape=(1,), keep_dims=True)
        input = torch.randn(6, 12, 10, 24)

        output = op(input, [])
        assert torch.equal(output, input)

        output = op(input, torch.tensor([1, 2]))
        assert output.shape == (6, 1, 1, 24)

        output = op(input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10, 1)


class TestReduceMinV1:
    @e2e_pytest_unit
    def test_forward(self):
        op = ReduceMinV1("dummy", shape=(1,), keep_dims=False)
        input = torch.randn(6, 12, 10, 24)

        output = op(input, [])
        assert torch.equal(output, input)

        output = op(input, torch.tensor([1, 2]))
        assert output.shape == (6, 24)

        output = op(input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10)

        op = ReduceMinV1("dummy", shape=(1,), keep_dims=True)
        input = torch.randn(6, 12, 10, 24)

        output = op(input, [])
        assert torch.equal(output, input)

        output = op(input, torch.tensor([1, 2]))
        assert output.shape == (6, 1, 1, 24)

        output = op(input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10, 1)


class TestReduceSumV1:
    @e2e_pytest_unit
    def test_forward(self):
        op = ReduceSumV1("dummy", shape=(1,), keep_dims=False)
        input = torch.randn(6, 12, 10, 24)

        output = op(input, [])
        assert torch.equal(output, input)

        output = op(input, torch.tensor([1, 2]))
        assert output.shape == (6, 24)
        assert torch.equal(output, torch.sum(input, dim=(1, 2)))

        output = op(input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10)
        assert torch.equal(output, torch.sum(input, dim=-1))

        op = ReduceSumV1("dummy", shape=(1,), keep_dims=True)
        input = torch.randn(6, 12, 10, 24)

        output = op(input, [])
        assert torch.equal(output, input)

        output = op(input, torch.tensor([1, 2]))
        assert output.shape == (6, 1, 1, 24)
        assert torch.equal(output, torch.sum(input, dim=(1, 2), keepdim=True))

        output = op(input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10, 1)
        assert torch.equal(output, torch.sum(input, dim=-1, keepdim=True))
