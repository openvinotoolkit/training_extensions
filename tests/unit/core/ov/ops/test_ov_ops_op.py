# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import openvino.runtime as ov

from otx.core.ov.ops.arithmetics import MultiplyV1
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOperation:
    @e2e_pytest_unit
    def test(self):
        data = ov.opset10.parameter([3, 1, 2], ov.Type.f32)
        mul_constant = ov.opset10.constant([1.5], ov.Type.f32)
        mul = ov.opset10.multiply(data, mul_constant)
        op = MultiplyV1.from_ov(mul)
        assert isinstance(op, MultiplyV1)
        assert op.type == MultiplyV1.TYPE
        assert op.version == MultiplyV1.VERSION
        assert op.name == op._name
        assert op.attrs == op._attrs
        assert op.shape == op.attrs.shape
        assert isinstance(repr(op), str)
