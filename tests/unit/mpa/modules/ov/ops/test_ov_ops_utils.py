# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

import openvino.runtime as ov
import pytest
import torch

from otx.core.ov.ops.movements import get_torch_padding
from otx.core.ov.ops.utils import get_dynamic_shape
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_get_dynamic_shape():
    op_ov = ov.opset10.parameter([-1, 256, 256, 3], ov.Type.f32)
    output = get_dynamic_shape(op_ov)
    assert output == [-1, 256, 256, 3]


@e2e_pytest_unit
def test_get_torch_padding():
    input = torch.randn(4, 3, 64, 64)

    output = get_torch_padding([5, 5], [5, 5], "valid", input.shape[2:], [3, 3], [1, 1])
    assert output == 0

    output = get_torch_padding([5, 5], [5, 5], "same_upper", input.shape[2:], [3, 3], [1, 1])
    output = output(input)
    assert output.shape == (4, 3, 66, 66)

    output = get_torch_padding([5, 5], [5, 5], "explicit", input.shape[2:], [3, 3], [1, 1])
    output = output(input)
    assert output.shape == (4, 3, 74, 74)

    with pytest.raises(NotImplementedError):
        get_torch_padding([5, 5], [5, 5], "error", input.shape[2:], [3, 3], [1, 1])
