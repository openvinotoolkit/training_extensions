# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.classification.adapters.mmcls.models.heads.mmov_cls_head import (
    MMOVClsHead,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.ov.models.mmcls.test_helpers import create_ov_model


class TestMMOVClsHead:
    @pytest.fixture(autouse=True)
    def setup(self):
        ov_model = create_ov_model()
        self.model = MMOVClsHead(
            model_path_or_model=ov_model,
        )

    @e2e_pytest_unit
    def test_forward_train(self):
        data = {}
        for key, shape in self.model.model.input_shapes.items():
            shape = [1 if i == -1 else i for i in shape]
            data[key] = torch.randn(shape)

        output = self.model.forward_train(list(data.values()), torch.tensor([0]))
        assert "loss" in output

    @e2e_pytest_unit
    def test_simple_test(self):
        data = {}
        for key, shape in self.model.model.input_shapes.items():
            shape = [1 if i == -1 else i for i in shape]
            data[key] = torch.randn(shape)

        outputs = self.model.simple_test(list(data.values()))
        assert isinstance(outputs, list)
        assert len(outputs) == 1
