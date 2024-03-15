# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from otx.algorithms.detection.adapters.mmdet.models.necks.mmov_ssd_neck import (
    MMOVSSDNeck,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMMOVSSDNeck:
    @pytest.fixture(autouse=True)
    def setup(self):

        param = ov.opset10.parameter([1, 24, 64, 64], ov.Type.f32, name="in")
        filter = ov.opset10.constant(np.random.normal(size=(1, 24, 64, 64)), ov.Type.f32)
        mul = ov.opset10.matmul(param, filter, False, False)
        result = ov.opset10.result(mul, name="out")

        ov_model = ov.Model([result], [param], "ssd_neck")

        self.model = MMOVSSDNeck(
            model_path_or_model=ov_model,
            inputs=dict(
                extra_layers=[
                    "in",
                ],
            ),
            outputs=dict(
                extra_layers=[
                    "out",
                ],
            ),
        )

    @e2e_pytest_unit
    def test_init_weights(self):
        self.model.init_weights()

    @e2e_pytest_unit
    def test_forward(self):
        data = {}
        for key, shape in self.model.extra_layers[0].input_shapes.items():
            shape = [1 if i == -1 else i for i in shape]
            data[key] = torch.randn(shape)
        self.model.train()
        self.model(list(data.values()))
