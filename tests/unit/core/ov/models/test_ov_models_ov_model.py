# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import openvino.runtime as ov
import torch

from otx.core.ov.models.ov_model import OVModel
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOVModel:
    @e2e_pytest_unit
    def test(self):

        param = ov.opset10.parameter([1, 3, 64, 64], ov.Type.f32, name="in")
        constant = ov.opset10.constant(np.array([103.0, 116.0, 123.0]).reshape(1, 3, 1, 1), ov.Type.f32)
        node = ov.opset10.subtract(param, constant, "numpy")
        constant = ov.opset10.constant(np.random.normal(size=(32, 3, 3, 3)), ov.Type.f32)
        node = ov.opset10.convolution(node, constant, [2, 2], [1, 1], [1, 1], [1, 1], "explicit")
        constant = ov.opset10.constant(np.random.normal(size=(1, 32, 1, 1)), ov.Type.f32)
        node = ov.opset10.add(node, constant, "numpy")
        node = ov.opset10.clamp(node, 0, 6)
        result = ov.opset10.result(node, name="out")
        ov_model = ov.Model([result], [param], "model")

        model = OVModel(
            model_path_or_model=ov_model,
            remove_normalize=True,
            init_weight=True,
            merge_bn=True,
            paired_bn=True,
        )
        assert model.inputs == model._inputs
        assert model.outputs == model._outputs
        assert model.features == model._feature_dict
        assert model.input_shapes == model._input_shapes
        assert model.output_shapes == model._output_shapes

        data = {}
        for key, shape in model.input_shapes.items():
            shape = [1 if i == -1 else i for i in shape]
            data[key] = torch.randn(shape)
        model(**data)
