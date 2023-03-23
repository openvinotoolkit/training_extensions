# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import pytest
import torch

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.modules.op_module import OperationModule
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOperationModule:
    @e2e_pytest_unit
    def test(sefl):

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
        with pytest.raises(ValueError):
            op(input_1=torch.tensor([1.0]))

        op = OperationModule(multiply, {"input_0": None, "input_1": constant})
        assert op.type == multiply.type
        assert op.version == multiply.version
        assert op.name == multiply.name
        assert op.shape == multiply.shape
        assert op.attrs == multiply.attrs
        assert torch.equal(op(torch.tensor([1.0])), torch.tensor([1.5]))
        assert torch.equal(op(input_0=torch.tensor([1.0])), torch.tensor([1.5]))
        with pytest.raises(ValueError):
            op(input_1=torch.tensor([1.0]))
