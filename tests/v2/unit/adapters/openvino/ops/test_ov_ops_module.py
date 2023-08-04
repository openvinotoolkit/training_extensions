# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch
from otx.v2.adapters.openvino.ops.builder import OPS
from otx.v2.adapters.openvino.ops.modules.op_module import OperationModule


class TestOperationModule:

    def test(self) -> None:

        constant_cls = OPS.get_by_name("ConstantV0")
        constant = constant_cls(
            "weight",
            data=torch.tensor([1.5]),
            shape=(1,),
            is_parameter=True,
        )

        multiply_cls = OPS.get_by_name("MultiplyV1")
        multiply = multiply_cls("multiply", shape=(1,))

        op = OperationModule(multiply, [None, constant])
        assert op.type == multiply.type
        assert op.version == multiply.version
        assert op.name == multiply.name
        assert op.shape == multiply.shape
        assert op.attrs == multiply.attrs
        assert torch.equal(op(torch.tensor([1.0])), torch.tensor([1.5]))
        assert torch.equal(op(input_0=torch.tensor([1.0])), torch.tensor([1.5]))
        with pytest.raises(ValueError, match="duplicated key"):
            op(input_1=torch.tensor([1.0]))

        op = OperationModule(multiply, {"input_0": None, "input_1": constant})
        assert op.type == multiply.type
        assert op.version == multiply.version
        assert op.name == multiply.name
        assert op.shape == multiply.shape
        assert op.attrs == multiply.attrs
        assert torch.equal(op(torch.tensor([1.0])), torch.tensor([1.5]))
        assert torch.equal(op(input_0=torch.tensor([1.0])), torch.tensor([1.5]))
        with pytest.raises(ValueError, match="duplicated key"):
            op(input_1=torch.tensor([1.0]))
