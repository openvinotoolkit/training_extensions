# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.backbones.mmov_backbone import (
    MMOVBackbone,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.ov.models.mmcls.test_helpers import create_ov_model


class TestMMOVBackbone:
    @pytest.fixture(autouse=True)
    def setup(self):
        ov_model = create_ov_model()
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
