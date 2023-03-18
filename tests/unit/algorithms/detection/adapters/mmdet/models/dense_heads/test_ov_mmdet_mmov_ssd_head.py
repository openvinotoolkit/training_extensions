# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from otx.algorithms.detection.adapters.mmdet.models.dense_heads.mmov_ssd_head import (
    MMOVSSDHead,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMMOVSSDHead:
    @pytest.fixture(autouse=True)
    def setup(self):

        params = []
        results = []

        param = ov.opset10.parameter([1, 24, 64, 64], ov.Type.f32, name="in")
        filter = ov.opset10.constant(np.random.normal(size=(1, 24, 64, 64)), ov.Type.f32)
        mul = ov.opset10.matmul(param, filter, False, False)
        result = ov.opset10.result(mul, name="reg_out")
        params.append(param)
        results.append(result)

        filter = ov.opset10.constant(np.random.normal(size=(1, 24, 64, 64)), ov.Type.f32)
        mul = ov.opset10.matmul(param, filter, False, False)
        result = ov.opset10.result(mul, name="cls_out")
        results.append(result)

        ov_model = ov.Model(results, params, "ssd_head")

        self.model = MMOVSSDHead(
            model_path_or_model=ov_model,
            transpose_reg=True,
            transpose_cls=True,
            background_index=0,
            inputs=dict(
                reg_convs=[
                    "in",
                ],
                cls_convs=[
                    "in",
                ],
            ),
            outputs=dict(
                reg_convs=[
                    "reg_out",
                ],
                cls_convs=[
                    "cls_out",
                ],
            ),
            num_classes=3,
            anchor_generator=dict(
                type="SSDAnchorGenerator",
                scale_major=False,
                input_size=300,
                basesize_ratio_range=(0.15, 0.9),
                strides=[24, 48, 92, 171, 400, 400],
                ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
        )

    @e2e_pytest_unit
    def test_init_weights(self):
        self.model.init_weights()

    @e2e_pytest_unit
    def test_forward(self):
        data = {}
        for key, shape in self.model.cls_convs[0].input_shapes.items():
            shape = [1 if i == -1 else i for i in shape]
            data[key] = torch.randn(shape)
        self.model.train()
        self.model(list(data.values()))
