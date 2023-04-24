# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import numpy as np
import openvino.runtime as ov


def create_ov_model():
    param = ov.opset10.parameter([1, 3, 32, 32], ov.Type.f32, name="in")
    constant = ov.opset10.constant(np.array([103.0, 116.0, 123.0]).reshape(1, 3, 1, 1), ov.Type.f32)
    node = ov.opset10.subtract(param, constant, "numpy")
    constant = ov.opset10.constant(np.random.normal(size=(16, 3, 3, 3)), ov.Type.f32)
    node = ov.opset10.convolution(node, constant, [2, 2], [1, 1], [1, 1], [1, 1], "explicit")
    constant = ov.opset10.constant(np.random.normal(size=(1, 16, 1, 1)), ov.Type.f32)
    node = ov.opset10.add(node, constant, "numpy")
    node = ov.opset10.clamp(node, 0, 6)
    node = ov.opset10.reduce_mean(node, [2, 3], False)
    node = ov.opset10.reshape(node, [-1, 16], False)
    constant = ov.opset10.constant(np.random.normal(size=(16, 10)), ov.Type.f32)
    node = ov.opset10.matmul(node, constant, False, False)
    result = ov.opset10.result(node, name="out")
    ov_model = ov.Model([result], [param], "model")
    return ov_model
