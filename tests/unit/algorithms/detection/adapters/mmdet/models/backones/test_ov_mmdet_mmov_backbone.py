# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import openvino.runtime as ov
import pytest
import torch

from otx.algorithms.detection.adapters.mmdet.models.backbones.mmov_backbone import (
    MMOVBackbone,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestMMOVBackbone:
    @pytest.fixture(autouse=True)
    def setup(self):

        param = ov.opset10.parameter([1, 3, 64, 64], ov.Type.f32, name="in")
        filter = ov.opset10.constant(np.random.normal(size=(1, 3, 64, 64)), ov.Type.f32)
        mul = ov.opset10.matmul(param, filter, False, False)
        result = ov.opset10.result(mul, name="out")
        ov_model = ov.Model([result], [param], "det_backbone")

        self.model = MMOVBackbone(
            model_path_or_model=ov_model,
            remove_normalize=True,
            merge_bn=True,
            paired_bn=True,
        )

    @e2e_pytest_unit
    def test_init_weights(self):
        self.model.init_weights()

    @e2e_pytest_unit
    def test_forward(self):
        assert self.model.inputs == self.model._inputs
        assert self.model.outputs == self.model._outputs
        assert self.model.features == self.model._feature_dict
        assert self.model.input_shapes == self.model._input_shapes
        assert self.model.output_shapes == self.model._output_shapes

        data = {}
        for key, shape in self.model.input_shapes.items():
            shape = [1 if i == -1 else i for i in shape]
            data[key] = torch.randn(shape)
        self.model.train()
        self.model(list(data.values()), torch.tensor([0]))
