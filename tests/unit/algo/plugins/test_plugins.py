# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test for otx.algo.plugins.xpu_precision"""


import pytest
import torch
from otx.algo.plugins.xpu_precision import MixedPrecisionXPUPlugin
from torch.optim import Optimizer


class TestMixedPrecisionXPUPlugin:
    @pytest.fixture()
    def plugin(self):
        return MixedPrecisionXPUPlugin()

    def test_init(self, plugin):
        assert plugin.scaler is None

    def test_pre_backward(self, plugin, mocker):
        tensor = torch.zeros(1)
        module = mocker.MagicMock()
        output = plugin.pre_backward(tensor, module)
        assert output == tensor

    def test_optimizer_step_no_scaler(self, plugin, mocker):
        optimizer = mocker.MagicMock(Optimizer)
        model = mocker.MagicMock()
        closure = mocker.MagicMock()
        kwargs = {}
        mock_optimizer_step = mocker.patch(
            "otx.algo.plugins.xpu_precision.Precision.optimizer_step",
        )
        out = plugin.optimizer_step(optimizer, model, closure, **kwargs)
        assert isinstance(out, mocker.MagicMock)
        mock_optimizer_step.assert_called_once()

    def test_optimizer_step_with_scaler(self, plugin, mocker):
        optimizer = mocker.MagicMock(Optimizer)
        model = mocker.MagicMock()
        closure = mocker.MagicMock()
        plugin.scaler = mocker.MagicMock()
        kwargs = {}
        out = plugin.optimizer_step(optimizer, model, closure, **kwargs)
        assert isinstance(out, mocker.MagicMock)

    def test_clip_gradients(self, plugin, mocker):
        optimizer = mocker.MagicMock(Optimizer)
        clip_val = 0.1
        gradient_clip_algorithm = "norm"
        mock_clip_gradients = mocker.patch(
            "otx.algo.plugins.xpu_precision.Precision.clip_gradients",
        )
        plugin.clip_gradients(optimizer, clip_val, gradient_clip_algorithm)
        mock_clip_gradients.assert_called_once()
