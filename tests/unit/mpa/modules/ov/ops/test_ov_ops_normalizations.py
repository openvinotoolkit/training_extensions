# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch
from torch.nn import functional as F

from otx.core.ov.ops.normalizations import (
    MVNV6,
    BatchNormalizationV0,
    LocalResponseNormalizationV0,
    NormalizeL2V0,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestBatchNormalizationV0:
    @e2e_pytest_unit
    def test_forward(self):
        op = BatchNormalizationV0("dummy", shape=(1,), epsilon=1e-05, max_init_iter=2)
        op.eval()
        input = torch.randn(8, 128, 256, 256)
        gamma = torch.randn(128)
        beta = torch.randn(128)
        mean = torch.zeros(128)
        variance = torch.ones(128)

        output = op(input, gamma, beta, mean, variance)
        ref = F.batch_norm(input, mean, variance, gamma, beta, eps=op.attrs.epsilon)
        assert torch.equal(output, ref)

        op.train()
        outputs = []
        for _ in range(op.attrs.max_init_iter + 1):
            outputs.append(op(input, gamma, beta, mean, variance))
        assert torch.equal(outputs[0], outputs[1])
        assert not torch.equal(outputs[1], outputs[2])


class TestLocalResponseNormalizationV0:
    @e2e_pytest_unit
    def test_forward(self):
        op = LocalResponseNormalizationV0(
            "dummy",
            shape=(1,),
            alpha=0.0001,
            beta=0.75,
            bias=1.0,
            size=2,
        )
        input = torch.randn(6, 12, 10, 24)
        output = op(input, torch.tensor([1]))
        ref = F.local_response_norm(input, size=2, alpha=0.0001, beta=0.75, k=1.0)
        assert torch.equal(output, ref)


class TestNormalizeL2V0:
    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            NormalizeL2V0("dummy", shape=(1,), eps=0.01, eps_mode="error")

    @e2e_pytest_unit
    def test_forward(self):
        op = NormalizeL2V0("dummy", shape=(1,), eps=0.01, eps_mode="add")
        input = torch.rand(6, 12, 10, 24)

        output = op(input, torch.tensor([]))
        ref = input / (input + op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(input, torch.tensor([1]))
        ref = input / (input.pow(2).sum(1, keepdim=True) + op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(input, torch.tensor([1, 2, 3]))
        ref = input / (input.pow(2).sum([1, 2, 3], keepdim=True) + op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(input, torch.tensor([-2, -3]))
        ref = input / (input.pow(2).sum([-2, -3], keepdim=True) + op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        op = NormalizeL2V0("dummy", shape=(1,), eps=0.01, eps_mode="max")
        input = torch.rand(6, 12, 10, 24)

        output = op(input, torch.tensor([]))
        ref = input / torch.clamp(input, max=op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(input, torch.tensor([1]))
        ref = input / torch.clamp(input.pow(2).sum(1, keepdim=True), max=op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(input, torch.tensor([1, 2, 3]))
        ref = input / torch.clamp(input.pow(2).sum([1, 2, 3], keepdim=True), max=op.attrs.eps).sqrt()
        assert torch.equal(output, ref)

        output = op(input, torch.tensor([-2, -3]))
        ref = input / torch.clamp(input.pow(2).sum([-2, -3], keepdim=True), max=op.attrs.eps).sqrt()
        assert torch.equal(output, ref)


class TestMVNV6:
    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            MVNV6("dummy", shape=(1,), normalize_variance=True, eps=0.01, eps_mode="error")

    @e2e_pytest_unit
    def test_forward(self):
        op = MVNV6(
            "dummy",
            shape=(1,),
            normalize_variance=False,
            eps=0.01,
            eps_mode="INSIDE_SQRT",
        )
        input = torch.randn(6, 12, 10, 24)
        output = op(input, torch.tensor([1, 2]))
        ref = input - input.mean([1, 2], keepdim=True)
        assert torch.equal(output, ref)

        op = MVNV6(
            "dummy",
            shape=(1,),
            normalize_variance=True,
            eps=0.01,
            eps_mode="INSIDE_SQRT",
        )
        input = torch.randn(6, 12, 10, 24)
        output = op(input, torch.tensor([1, 2]))
        ref = input - input.mean([1, 2], keepdim=True)
        ref = ref / torch.sqrt(torch.square(ref).mean([1, 2], keepdim=True) + 0.01)
        assert torch.equal(output, ref)

        op = MVNV6(
            "dummy",
            shape=(1,),
            normalize_variance=True,
            eps=0.01,
            eps_mode="OUTSIDE_SQRT",
        )
        input = torch.randn(6, 12, 10, 24)
        output = op(input, torch.tensor([1, 2]))
        ref = input - input.mean([1, 2], keepdim=True)
        ref = ref / (torch.sqrt(torch.square(ref).mean([1, 2], keepdim=True)) + 0.01)
        assert torch.equal(output, ref)
