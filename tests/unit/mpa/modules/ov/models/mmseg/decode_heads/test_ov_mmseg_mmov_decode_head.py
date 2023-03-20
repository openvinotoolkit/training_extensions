# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from otx.algorithms.segmentation.adapters.mmseg.models import MMOVDecodeHead
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMMOVBackbone:
    @pytest.fixture(autouse=True)
    def setup(self):

        params = []
        results = []

        param = ov.opset10.parameter([1, 24, 64, 64], ov.Type.f32, name="in")
        filter = ov.opset10.constant(np.random.normal(size=(1, 24, 64, 64)), ov.Type.f32)
        mul = ov.opset10.matmul(param, filter, False, False)
        result = ov.opset10.result(mul, name="extractor_out")
        params.append(param)
        results.append(result)

        filter = ov.opset10.constant(np.random.normal(size=(1, 24, 64, 64)), ov.Type.f32)
        mul = ov.opset10.matmul(param, filter, False, False)
        result = ov.opset10.result(mul, name="cls_seg_out")
        results.append(result)

        ov_model = ov.Model(results, params, "seg_head")

        self.model = MMOVDecodeHead(
            model_path_or_model=ov_model,
            inputs=dict(
                extractor="in",
                cls_seg="in",
            ),
            outputs=dict(
                extractor="extractor_out",
                cls_seg="cls_seg_out",
            ),
            in_channels=320,
            num_classes=24,
        )

    @e2e_pytest_unit
    def test_init_weights(self):
        self.model.init_weights()

    @e2e_pytest_unit
    def test_forward(self):

        data = {}
        input_shapes = self.model.conv_seg.input_shapes
        if getattr(self.model, "extractor", None):
            input_shapes = self.model.extractor.input_shapes

        for key, shape in input_shapes.items():
            shape = [1 if i == -1 else i for i in shape]
            data[key] = torch.randn(shape)

        output = self.model(list(data.values()))
        assert output.shape[1] == 24
