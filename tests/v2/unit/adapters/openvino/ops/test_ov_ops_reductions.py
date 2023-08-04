# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from otx.v2.adapters.openvino.ops.reductions import (
    ReduceMeanV1,
    ReduceMinV1,
    ReduceProdV1,
    ReduceSumV1,
)


class TestReduceMeanV1:

    def test_forward(self) -> None:
        op = ReduceMeanV1("dummy", shape=(1,), keep_dims=False)
        _input = torch.randn(6, 12, 10, 24)

        output = op(_input, [])
        assert torch.equal(output, _input)

        output = op(_input, torch.tensor([1, 2]))
        assert output.shape == (6, 24)
        assert torch.equal(output, torch.mean(_input, dim=(1, 2)))

        output = op(_input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10)
        assert torch.equal(output, torch.mean(_input, dim=-1))

        op = ReduceMeanV1("dummy", shape=(1,), keep_dims=True)

        output = op(_input, [])
        assert torch.equal(output, _input)

        output = op(_input, torch.tensor([1, 2]))
        assert output.shape == (6, 1, 1, 24)
        assert torch.equal(output, torch.mean(_input, dim=(1, 2), keepdim=True))

        output = op(_input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10, 1)
        assert torch.equal(output, torch.mean(_input, dim=-1, keepdim=True))


class TestReduceProdV1:

    def test_forward(self) -> None:
        op = ReduceProdV1("dummy", shape=(1,), keep_dims=False)
        _input = torch.randn(6, 12, 10, 24)

        output = op(_input, [])
        assert torch.equal(output, _input)

        output = op(_input, torch.tensor([1, 2]))
        assert output.shape == (6, 24)

        output = op(_input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10)

        op = ReduceProdV1("dummy", shape=(1,), keep_dims=True)
        _input = torch.randn(6, 12, 10, 24)

        output = op(_input, [])
        assert torch.equal(output, _input)

        output = op(_input, torch.tensor([1, 2]))
        assert output.shape == (6, 1, 1, 24)

        output = op(_input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10, 1)


class TestReduceMinV1:

    def test_forward(self) -> None:
        op = ReduceMinV1("dummy", shape=(1,), keep_dims=False)
        _input = torch.randn(6, 12, 10, 24)

        output = op(_input, [])
        assert torch.equal(output, _input)

        output = op(_input, torch.tensor([1, 2]))
        assert output.shape == (6, 24)

        output = op(_input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10)

        op = ReduceMinV1("dummy", shape=(1,), keep_dims=True)
        _input = torch.randn(6, 12, 10, 24)

        output = op(_input, [])
        assert torch.equal(output, _input)

        output = op(_input, torch.tensor([1, 2]))
        assert output.shape == (6, 1, 1, 24)

        output = op(_input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10, 1)


class TestReduceSumV1:

    def test_forward(self) -> None:
        op = ReduceSumV1("dummy", shape=(1,), keep_dims=False)
        _input = torch.randn(6, 12, 10, 24)

        output = op(_input, [])
        assert torch.equal(output, _input)

        output = op(_input, torch.tensor([1, 2]))
        assert output.shape == (6, 24)
        assert torch.equal(output, torch.sum(_input, dim=(1, 2)))

        output = op(_input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10)
        assert torch.equal(output, torch.sum(_input, dim=-1))

        op = ReduceSumV1("dummy", shape=(1,), keep_dims=True)
        _input = torch.randn(6, 12, 10, 24)

        output = op(_input, [])
        assert torch.equal(output, _input)

        output = op(_input, torch.tensor([1, 2]))
        assert output.shape == (6, 1, 1, 24)
        assert torch.equal(output, torch.sum(_input, dim=(1, 2), keepdim=True))

        output = op(_input, torch.tensor([-1]))
        assert output.shape == (6, 12, 10, 1)
        assert torch.equal(output, torch.sum(_input, dim=-1, keepdim=True))
