# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import openvino.runtime as ov
import pytest
import torch
from otx.v2.adapters.openvino.ops.infrastructures import ConstantV0, ParameterV0, ResultV0


class TestParameterV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)


    def test_invalid_attr(self) -> None:
        with pytest.raises(ValueError, match="Invalid element_type error."):
            ParameterV0("dummy", shape=(self.shape,), element_type="error")


    def test_forward(self) -> None:
        op = ParameterV0("dummy", shape=(self.shape,))
        assert torch.equal(self.input, op(self.input))

        op = ParameterV0("dummy", shape=(self.shape,), permute=(0, 2, 3, 1))
        assert torch.equal(self.input.permute(0, 2, 3, 1), op(self.input))

        op = ParameterV0("dummy", shape=(self.shape,), verify_shape=True)
        assert torch.equal(self.input, op(self.input))

        op = ParameterV0("dummy", shape=((-1, 3, -1, -1),), verify_shape=True)
        assert torch.equal(self.input, op(self.input))


    def test_from_ov(self) -> None:
        op_ov = ov.opset10.parameter([-1, 256, 256, 3], ov.Type.f32)
        op_ov.set_layout(ov.Layout("NHWC"))
        op = ParameterV0.from_ov(op_ov)

        assert isinstance(op, ParameterV0)
        assert op.attrs.element_type == "f32"
        assert op.attrs.layout == ("N", "C", "H", "W")
        assert op.attrs.permute == (0, 2, 3, 1)
        assert op.attrs.shape == ((-1, 3, 256, 256),)


class TestResultV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (4, 3, 256, 256)
        self.input = torch.randn(self.shape)


    def test_forward(self) -> None:
        op = ResultV0("dummy", shape=(self.shape,))
        assert torch.equal(self.input, op(self.input))


class TestConstantV0:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.shape = (512, 256)
        self.data = torch.randn(self.shape)


    def test_invalid_attr(self) -> None:
        with pytest.raises(KeyError):
            ConstantV0("dummy", shape=(self.shape,))


    def test_forward(self) -> None:
        op = ConstantV0("dummy", data=self.data, shape=(self.shape,), is_parameter=False)
        assert isinstance(op.data, torch.Tensor)
        assert torch.equal(self.data, op())

        op = ConstantV0("dummy", data=self.data, shape=(self.shape,), is_parameter=True)
        assert isinstance(op.data, torch.nn.parameter.Parameter)
        assert torch.equal(self.data, op())


    def test_from_ov(self) -> None:
        op_ov = ov.opset10.constant(self.data.numpy().astype(np.uint64), ov.Type.u64)
        op = ConstantV0.from_ov(op_ov)
        assert isinstance(op, ConstantV0)
        assert op.attrs.shape == self.shape
        assert not op.attrs.is_parameter
        assert isinstance(op.data, torch.Tensor)

        op_ov = ov.opset10.constant(self.data.numpy(), ov.Type.f32)
        op = ConstantV0.from_ov(op_ov)
        assert isinstance(op, ConstantV0)
        assert op.attrs.shape == self.shape
        assert not op.attrs.is_parameter
        assert isinstance(op.data, torch.Tensor)

        op_ov = ov.opset10.constant(self.data.numpy(), ov.Type.f32)
        data = ov.opset10.parameter([4, 512], ov.Type.f32)
        ov.opset10.matmul(data, op_ov, False, False)
        op = ConstantV0.from_ov(op_ov)
        assert isinstance(op, ConstantV0)
        assert op.attrs.shape == self.shape
        assert op.attrs.is_parameter
        assert isinstance(op.data, torch.nn.parameter.Parameter)
