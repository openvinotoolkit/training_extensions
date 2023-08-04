# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino.runtime as ov
from otx.v2.adapters.openvino.ops.arithmetics import MultiplyV1


class TestOperation:

    def test(self) -> None:
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
