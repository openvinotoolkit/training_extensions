# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from otx.algorithms.detection.adapters.mmdet.models.dense_heads.mmov_rpn_head import (
    MMOVRPNHead,
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
        result_1 = ov.opset10.result(mul, name="cls_score_out")
        result_2 = ov.opset10.result(mul, name="bbox_pred_out")
        params.append(param)
        results.append(result_1)
        results.append(result_2)

        ov_model = ov.Model(results, params, "rpn_head")

        self.model = MMOVRPNHead(
            model_path_or_model=ov_model,
            transpose_reg=True,
            transpose_cls=True,
            inputs="in",
            outputs=[
                "cls_score_out",
                "bbox_pred_out",
            ],
            anchor_generator=dict(
                type="AnchorGenerator",
                base_sizes=[256],
                scales=[0.25, 0.5, 1, 2],
                ratios=[0.5, 1.0, 2.0],
                strides=[8],
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
        for key, shape in self.model.model.input_shapes.items():
            shape = [1 if i == -1 else i for i in shape]
            data[key] = torch.randn(shape)
        self.model.train()
        self.model(list(data.values()))
