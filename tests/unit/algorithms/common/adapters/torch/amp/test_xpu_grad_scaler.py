"""Test for otx.algorithms.common.adapters.torch.amp.xpu_grad_scaler """

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from otx.algorithms.common.adapters.torch.amp.xpu_grad_scaler import XPUGradScaler


class TestXPUGradScaler:
    @pytest.fixture
    def grad_scaler(self, mocker):
        mocker.patch("otx.algorithms.common.adapters.torch.amp.xpu_grad_scaler.is_xpu_available", return_value=True)
        return XPUGradScaler()

    @pytest.fixture
    def optimizer(self):
        model = torch.nn.Linear(3, 3)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return optimizer

    def test_init(self, grad_scaler):
        assert grad_scaler._enabled
        assert grad_scaler._init_scale == 2.0**16
        assert grad_scaler._growth_factor == 2.0
        assert grad_scaler._backoff_factor == 0.5
        assert grad_scaler._growth_interval == 2000

    def test_scale(self, grad_scaler, mocker):
        outputs = mocker.MagicMock(torch.Tensor)
        outputs.device.type = "xpu"
        outputs.device.index = 0
        grad_scaler._lazy_init_scale_growth_tracker = mocker.MagicMock()
        grad_scaler._scale = mocker.MagicMock()
        scaled_outputs = grad_scaler.scale(outputs)
        assert isinstance(scaled_outputs.device.type, mocker.MagicMock)
