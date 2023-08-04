# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.v2.adapters.openvino.ops.normalizations import (
    MVNV6,
    BatchNormalizationV0,
    LocalResponseNormalizationV0,
    NormalizeL2V0,
)
from torch.nn import functional


class TestBatchNormalizationV0:

    def test_forward(self) -> None:
        op = BatchNormalizationV0("dummy", shape=(1,), epsilon=1e-05, max_init_iter=2)
        op.eval()
        _input = torch.randn(8, 128, 256, 256)
        gamma = torch.randn(128)
        beta = torch.randn(128)
        mean = torch.zeros(128)
        variance = torch.ones(128)

        output = op(_input, gamma, beta, mean, variance)
        ref = functional.batch_norm(_input, mean, variance, gamma, beta, eps=op.attrs.epsilon)
        assert torch.equal(output, ref)

        op.train()
        outputs = []
        for _ in range(op.attrs.max_init_iter + 1):
            outputs.append(op(_input, gamma, beta, mean, variance))
        assert torch.equal(outputs[0], outputs[1])
        assert not torch.equal(outputs[1], outputs[2])


class TestLocalResponseNormalizationV0:

    def test_forward(self) -> None:
        op = LocalResponseNormalizationV0(
            "dummy",
            shape=(1,),
            alpha=0.0001,
            beta=0.75,
            bias=1.0,
            size=2,
        )
        _input = torch.randn(6, 12, 10, 24)
        output = op(_input, torch.tensor([1]))
        ref = functional.local_response_norm(_input, size=2, alpha=0.0001, beta=0.75, k=1.0)
        assert torch.equal(output, ref)


class TestNormalizeL2V0:

    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid eps_mode error."):
            NormalizeL2V0("dummy", shape=(1,), eps=0.01, eps_mode="error")


    def test_forward(self) -> None:
        op = NormalizeL2V0("dummy", shape=(1,), eps=0.01, eps_mode="add")
        _input = torch.rand(6, 12, 10, 24)

        output = op(_input, torch.tensor([]))
        ref = _input / (_input + op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(_input, torch.tensor([1]))
        ref = _input / (_input.pow(2).sum(1, keepdim=True) + op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(_input, torch.tensor([1, 2, 3]))
        ref = _input / (_input.pow(2).sum([1, 2, 3], keepdim=True) + op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(_input, torch.tensor([-2, -3]))
        ref = _input / (_input.pow(2).sum([-2, -3], keepdim=True) + op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        op = NormalizeL2V0("dummy", shape=(1,), eps=0.01, eps_mode="max")
        _input = torch.rand(6, 12, 10, 24)

        output = op(_input, torch.tensor([]))
        ref = _input / torch.clamp(_input, max=op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(_input, torch.tensor([1]))
        ref = _input / torch.clamp(_input.pow(2).sum(1, keepdim=True), max=op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(_input, torch.tensor([1, 2, 3]))
        ref = _input / torch.clamp(_input.pow(2).sum([1, 2, 3], keepdim=True), max=op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(_input, torch.tensor([-2, -3]))
        ref = _input / torch.clamp(_input.pow(2).sum([-2, -3], keepdim=True), max=op.attrs.eps).sqrt()
        assert torch.equal(output, ref)


class TestMVNV6:

    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid eps_mode error."):
            MVNV6("dummy", shape=(1,), normalize_variance=True, eps=0.01, eps_mode="error")


    def test_forward(self) -> None:
        op = MVNV6(
            "dummy",
            shape=(1,),
            normalize_variance=False,
            eps=0.01,
            eps_mode="INSIDE_SQRT",
        )
        _input = torch.randn(6, 12, 10, 24)
        output = op(_input, torch.tensor([1, 2]))
        ref = _input - _input.mean([1, 2], keepdim=True)
        assert torch.equal(output, ref)

        op = MVNV6(
            "dummy",
            shape=(1,),
            normalize_variance=True,
            eps=0.01,
            eps_mode="INSIDE_SQRT",
        )
        _input = torch.randn(6, 12, 10, 24)
        output = op(_input, torch.tensor([1, 2]))
        ref = _input - _input.mean([1, 2], keepdim=True)
        ref = ref / torch.sqrt(torch.square(ref).mean([1, 2], keepdim=True) + 0.01)
        assert torch.equal(output, ref)

        op = MVNV6(
            "dummy",
            shape=(1,),
            normalize_variance=True,
            eps=0.01,
            eps_mode="OUTSIDE_SQRT",
        )
        _input = torch.randn(6, 12, 10, 24)
        output = op(_input, torch.tensor([1, 2]))
        ref = _input - _input.mean([1, 2], keepdim=True)
        ref = ref / (torch.sqrt(torch.square(ref).mean([1, 2], keepdim=True)) + 0.01)
        assert torch.equal(output, ref)
