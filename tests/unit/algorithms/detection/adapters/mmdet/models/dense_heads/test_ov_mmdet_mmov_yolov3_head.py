# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from otx.algorithms.detection.adapters.mmdet.models.dense_heads.mmov_yolov3_head import (
    MMOVYOLOV3Head,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMMOVYOLOV3Head:
    @pytest.fixture(autouse=True)
    def setup(self):

        params = []
        results = []

        param = ov.opset10.parameter([1, 24, 64, 64], ov.Type.f32, name="in")
        params.append(param)
        constant = ov.opset10.constant(np.random.normal(size=(32, 24, 3, 3)), ov.Type.f32)
        node = ov.opset10.convolution(param, constant, [2, 2], [1, 1], [1, 1], [1, 1], "explicit", name="pred_in_1_tmp")
        result = ov.opset10.result(node, name="bridge_out_1")
        results.append(result)
        constant = ov.opset10.constant(np.random.normal(size=(32, 32, 3, 3)), ov.Type.f32)
        node = ov.opset10.convolution(node, constant, [2, 2], [1, 1], [1, 1], [1, 1], "explicit", name="pred_in_1")
        result = ov.opset10.result(node, name="pred_out_1")
        results.append(result)

        constant = ov.opset10.constant(np.random.normal(size=(16, 24, 3, 3)), ov.Type.f32)
        node = ov.opset10.convolution(param, constant, [2, 2], [1, 1], [1, 1], [1, 1], "explicit", name="pred_in_2_tmp")
        result = ov.opset10.result(node, name="bridge_out_2")
        results.append(result)
        constant = ov.opset10.constant(np.random.normal(size=(16, 16, 3, 3)), ov.Type.f32)
        node = ov.opset10.convolution(node, constant, [2, 2], [1, 1], [1, 1], [1, 1], "explicit", name="pred_in_2")
        result = ov.opset10.result(node, name="pred_out_2")
        results.append(result)

        ov_model = ov.Model(results, params, "yolov3_head")
        with tempfile.TemporaryDirectory() as tempdir:
            ov.serialize(ov_model, os.path.join(tempdir, "model.xml"), os.path.join(tempdir, "model.bin"))

            self.model = MMOVYOLOV3Head(
                model_path_or_model=os.path.join(tempdir, "model.xml"),
                inputs=dict(
                    convs_bridge=[
                        "in",
                        "in",
                    ],
                    convs_pred=[
                        "pred_in_1_tmp||pred_in_1",
                        "pred_in_2_tmp||pred_in_2",
                    ],
                ),
                outputs=dict(
                    convs_bridge=[
                        "bridge_out_1",
                        "bridge_out_2",
                    ],
                    convs_pred=[
                        "pred_out_1",
                        "pred_out_2",
                    ],
                ),
                anchor_generator=dict(
                    type="YOLOAnchorGenerator",
                    base_sizes=[[(81, 82), (135, 169), (344, 319)], [(23, 27), (37, 58), (81, 82)]],
                    strides=[32, 16],
                ),
                bbox_coder=dict(type="YOLOBBoxCoder"),
                featmap_strides=[32, 16],
                num_classes=80,
            )

    @e2e_pytest_unit
    def test_init_weights(self):
        self.model.init_weights()

    @e2e_pytest_unit
    def test_forward(self):
        data = []
        for conv_bridge in self.model.convs_bridge:
            for key, shape in conv_bridge.input_shapes.items():
                shape = [1 if i == -1 else i for i in shape]
                data.append(torch.randn(shape))
        self.model.train()
        self.model(data)
