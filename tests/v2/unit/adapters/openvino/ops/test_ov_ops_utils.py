# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino.runtime as ov
import pytest
import torch
from otx.v2.adapters.openvino.ops.movements import get_torch_padding
from otx.v2.adapters.openvino.ops.utils import get_dynamic_shape


def test_get_dynamic_shape() -> None:
    op_ov = ov.opset10.parameter([-1, 256, 256, 3], ov.Type.f32)
    output = get_dynamic_shape(op_ov)
    assert output == [-1, 256, 256, 3]


def test_get_torch_padding() -> None:
    _input = torch.randn(4, 3, 64, 64)

    output = get_torch_padding([5, 5], [5, 5], "valid", _input.shape[2:], [3, 3], [1, 1])
    assert output == 0

    output = get_torch_padding([5, 5], [5, 5], "same_upper", _input.shape[2:], [3, 3], [1, 1])
    output = output(_input)
    assert output.shape == (4, 3, 66, 66)

    output = get_torch_padding([5, 5], [5, 5], "explicit", _input.shape[2:], [3, 3], [1, 1])
    output = output(_input)
    assert output.shape == (4, 3, 74, 74)

    with pytest.raises(NotImplementedError):
        get_torch_padding([5, 5], [5, 5], "error", _input.shape[2:], [3, 3], [1, 1])
