# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import tempfile

import openvino.runtime as ov
import pytest

from otx.core.ov.omz_wrapper import get_omz_model
from otx.core.ov.ops.infrastructures import ParameterV0
from otx.core.ov.ops.modules.op_module import convert_op_to_torch_module
from otx.core.ov.ops.utils import convert_op_to_torch
from otx.core.ov.utils import (
    get_op_name,
    load_ov_model,
    normalize_name,
    unnormalize_name,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


@e2e_pytest_unit
def test_load_ov_model():
    with tempfile.TemporaryDirectory() as tempdir:
        model = get_omz_model("mobilenet-v2-pytorch", download_dir=tempdir, output_dir=tempdir)
        model = load_ov_model(**model)

        model = load_ov_model("omz://mobilenet-v2-pytorch")
        assert not model.inputs[0].get_partial_shape().is_dynamic

        model = load_ov_model("omz://mobilenet-v2-pytorch", convert_dynamic=True)
        assert model.inputs[0].get_partial_shape().is_dynamic


@e2e_pytest_unit
def test_normalize_name():
    assert "dummy" == normalize_name("dummy")
    assert "dummy#dummy" == normalize_name("dummy.dummy")
    assert "dummy###dummy" == normalize_name("dummy...dummy")
    assert "dummy#dummy#dummy" == normalize_name("dummy.dummy.dummy")


@e2e_pytest_unit
def test_unnormalize_name():
    assert "dummy" == unnormalize_name("dummy")
    assert "dummy.dummy" == unnormalize_name("dummy#dummy")
    assert "dummy...dummy" == unnormalize_name("dummy###dummy")
    assert "dummy.dummy.dummy" == unnormalize_name("dummy#dummy#dummy")


@e2e_pytest_unit
def test_get_op_name():
    assert "dummy" == get_op_name(ov.opset10.parameter([3, 1, 2], ov.Type.f32, name="dummy"))
    assert "dummy#dummy" == get_op_name(ov.opset10.parameter([3, 1, 2], ov.Type.f32, name="dummy.dummy"))


@e2e_pytest_unit
def test_convert_op_to_torch():
    dummy = ov.opset10.parameter([3, 1, 2], ov.Type.f32)
    assert isinstance(convert_op_to_torch(dummy), ParameterV0)

    dummy = ov.opset10.depth_to_space(dummy, mode="blocks_first")
    with pytest.raises(KeyError):
        convert_op_to_torch(dummy)


@e2e_pytest_unit
def test_convert_op_to_torch_module():
    data = ov.opset10.parameter([3, 1, 2], ov.Type.f32)
    mul_constant = ov.opset10.constant([1.5], ov.Type.f32, name="weight")
    mul = ov.opset10.multiply(data, mul_constant)

    module = convert_op_to_torch_module(mul)
    should_none, node = list(module._dependent_ops.values())
    assert should_none is None
    assert node is not None and node.name == "weight"
