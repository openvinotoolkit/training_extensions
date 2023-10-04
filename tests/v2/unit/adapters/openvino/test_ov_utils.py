# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import tempfile

import openvino.runtime as ov
import pytest
from otx.v2.adapters.openvino.omz_wrapper import get_omz_model
from otx.v2.adapters.openvino.ops.infrastructures import ParameterV0
from otx.v2.adapters.openvino.ops.modules.op_module import convert_op_to_torch_module
from otx.v2.adapters.openvino.ops.utils import convert_op_to_torch
from otx.v2.adapters.openvino.utils import (
    get_op_name,
    load_ov_model,
    normalize_name,
    unnormalize_name,
)


def test_load_ov_model() -> None:
    with tempfile.TemporaryDirectory() as tempdir:
        model = get_omz_model("mobilenet-v2-pytorch", download_dir=tempdir, output_dir=tempdir)
        model = load_ov_model(**model)

        model = load_ov_model("omz://mobilenet-v2-pytorch")
        assert not model.inputs[0].get_partial_shape().is_dynamic

        model = load_ov_model("omz://mobilenet-v2-pytorch", convert_dynamic=True)
        assert model.inputs[0].get_partial_shape().is_dynamic


def test_normalize_name() -> None:
    assert normalize_name("dummy") == "dummy"
    assert normalize_name("dummy.dummy") == "dummy#dummy"
    assert normalize_name("dummy...dummy") == "dummy###dummy"
    assert normalize_name("dummy.dummy.dummy") == "dummy#dummy#dummy"


def test_unnormalize_name() -> None:
    assert unnormalize_name("dummy") == "dummy"
    assert unnormalize_name("dummy#dummy") == "dummy.dummy"
    assert unnormalize_name("dummy###dummy") == "dummy...dummy"
    assert unnormalize_name("dummy#dummy#dummy") == "dummy.dummy.dummy"


def test_get_op_name() -> None:
    assert get_op_name(ov.opset10.parameter([3, 1, 2], ov.Type.f32, name="dummy")) == "dummy"
    assert get_op_name(ov.opset10.parameter([3, 1, 2], ov.Type.f32, name="dummy.dummy")) == "dummy#dummy"


def test_convert_op_to_torch() -> None:
    dummy = ov.opset10.parameter([3, 1, 2], ov.Type.f32)
    assert isinstance(convert_op_to_torch(dummy), ParameterV0)

    dummy = ov.opset10.depth_to_space(dummy, mode="blocks_first")
    with pytest.raises(KeyError):
        convert_op_to_torch(dummy)


def test_convert_op_to_torch_module() -> None:
    data = ov.opset10.parameter([3, 1, 2], ov.Type.f32)
    mul_constant = ov.opset10.constant([1.5], ov.Type.f32, name="weight")
    mul = ov.opset10.multiply(data, mul_constant)

    module = convert_op_to_torch_module(mul)
    should_none, node = list(module._dependent_ops.values())
    assert should_none is None
    assert node is not None
    assert node.name == "weight"
