# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch

from otx.core.ov.ops.shape_manipulations import (
    ReshapeV1,
    ShapeOfV0,
    ShapeOfV3,
    SqueezeV0,
    UnsqueezeV0,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestSqueezeV0:
    @e2e_pytest_unit
    def test_forward(self):
        op = SqueezeV0("dummy", shape=(1,))
        input = torch.randn(1, 1, 5, 1, 1)

        output = op(input)
        assert output.shape == (5,)

        output = op(input, torch.tensor(0))
        assert output.shape == (1, 5, 1, 1)

        output = op(input, torch.tensor([0, 1]))
        assert output.shape == (5, 1, 1)

        output = op(input, torch.tensor([-2]))
        assert output.shape == (1, 1, 5, 1)


class TestUnsqueezeV0:
    @e2e_pytest_unit
    def test_forward(self):
        op = UnsqueezeV0("dummy", shape=(1,))
        input = torch.randn(2, 3)

        output = op(input, torch.tensor(2))
        assert output.shape == (2, 3, 1)

        output = op(input, torch.tensor([0, 1]))
        assert output.shape == (1, 2, 1, 3)

        output = op(input, torch.tensor([-1]))
        assert output.shape == (2, 3, 1)

        output = op(input, torch.tensor([-1, -2]))
        assert output.shape == (1, 2, 1, 3)


class TestReshapeV1:
    @e2e_pytest_unit
    def test_forward(self):
        op = ReshapeV1("dummy", shape=(1,), special_zero=False)
        input = torch.randn(2, 5, 5, 0)
        output = op(input, torch.tensor((0, 4)))
        assert output.shape == (0, 4)

        op = ReshapeV1("dummy", shape=(1,), special_zero=True)
        input = torch.randn(2, 5, 5, 24)
        output = op(input, torch.tensor((0, -1, 4)))
        assert output.shape == (2, 150, 4)

        op = ReshapeV1("dummy", shape=(1,), special_zero=True)
        input = torch.randn(2, 2, 3)
        output = op(input, torch.tensor((0, 0, 1, -1)))
        assert output.shape == (2, 2, 1, 3)

        op = ReshapeV1("dummy", shape=(1,), special_zero=True)
        input = torch.randn(3, 1, 1)
        output = op(input, torch.tensor((0, -1)))
        assert output.shape == (3, 1)


class TestShapeOfV0:
    @e2e_pytest_unit
    def test_forward(self):
        op = ShapeOfV0("dummy", shape=(1,))
        assert torch.equal(op(torch.randn(5, 3)), torch.tensor([5, 3]))


class TestShapeOfV3:
    @e2e_pytest_unit
    def test_invalid_attr(self):
        with pytest.raises(ValueError):
            ShapeOfV3("dummy", shape=(1,), output_type="error")

    @e2e_pytest_unit
    def test_forward(self):
        op = ShapeOfV3("dummy", shape=(1,), output_type="i32")
        output = op(torch.randn(5, 3))
        assert torch.equal(output, torch.tensor([5, 3]))
        assert output.dtype == torch.int32

        op = ShapeOfV3("dummy", shape=(1,), output_type="i64")
        output = op(torch.randn(5, 3))
        assert torch.equal(output, torch.tensor([5, 3]))
        assert output.dtype == torch.int64
