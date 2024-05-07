# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test for otx.algo.plugins.xpu_precision"""


from unittest.mock import MagicMock

import pytest
import torch
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from otx.algo.plugins import xpu_precision as target_file
from otx.algo.plugins.xpu_precision import MixedPrecisionXPUPlugin
from torch.optim import LBFGS


class TestMixedPrecisionXPUPlugin:
    def test_init(self):
        plugin = MixedPrecisionXPUPlugin()
        assert plugin.scaler is None

    @pytest.fixture()
    def mock_scaler_step_output(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def mock_scaler_state_dict(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture()
    def mock_scaler(self, mock_scaler_step_output, mock_scaler_state_dict) -> MagicMock:
        scaler = MagicMock()
        scaler.scale.side_effect = lambda x: x
        scaler.step.return_value = mock_scaler_step_output
        scaler.state_dict.return_value = mock_scaler_state_dict
        return scaler

    def test_init_w_scaler(self, mock_scaler):
        plugin = MixedPrecisionXPUPlugin(mock_scaler)
        assert plugin.scaler == mock_scaler

    @pytest.fixture()
    def plugin(self, mock_scaler) -> MixedPrecisionXPUPlugin:
        return MixedPrecisionXPUPlugin(mock_scaler)

    @pytest.fixture()
    def mock_lgt_module(self) -> MagicMock:
        return MagicMock()

    def test_pre_backward(self, plugin, mock_lgt_module, mock_scaler):
        tensor = torch.zeros(1)
        output = plugin.pre_backward(tensor, mock_lgt_module)

        mock_scaler.scale.assert_called_once_with(tensor)
        assert output == tensor

    @pytest.fixture()
    def mock_optimizer(self) -> MagicMock:
        optimizer = MagicMock()
        optimizer._step_supports_amp_scaling = False
        return optimizer

    @pytest.fixture()
    def mock_model(self) -> MagicMock:
        model = MagicMock()
        model.automatic_optimization = True
        return model

    @pytest.fixture()
    def mock_closure(self) -> MagicMock:
        return MagicMock(return_value=1)

    def test_optimizer_step(
        self,
        plugin,
        mock_scaler,
        mock_optimizer,
        mock_model,
        mock_closure,
        mock_scaler_step_output,
    ):
        output = plugin.optimizer_step(mock_optimizer, mock_model, mock_closure)

        mock_closure.assert_called_once()
        mock_scaler.unscale_.assert_called_once_with(mock_optimizer)
        mock_scaler.step.assert_called_once_with(mock_optimizer)
        mock_scaler.update.assert_called_once()
        assert output == mock_scaler_step_output

    def test_optimizer_step_manual_opt(
        self,
        plugin,
        mock_scaler,
        mock_optimizer,
        mock_model,
        mock_closure,
    ):
        mock_closure.return_value = None
        output = plugin.optimizer_step(mock_optimizer, mock_model, mock_closure)

        mock_closure.assert_called_once()
        mock_scaler.unscale_.assert_called_once_with(mock_optimizer)
        mock_scaler.step.assert_not_called()
        mock_scaler.update.assert_not_called()
        assert output is None

    def test_optimizer_step_lbfgs_optim(
        self,
        plugin,
        mock_model,
        mock_closure,
    ):
        optimizer = MagicMock(spec=LBFGS)
        with pytest.raises(MisconfigurationException, match="Native AMP and the LBFGS optimizer are not compatible."):
            plugin.optimizer_step(optimizer, mock_model, mock_closure)

    def test_optimizer_step_no_scaler(
        self,
        mock_optimizer,
        mock_model,
        mock_closure,
    ):
        plugin = MixedPrecisionXPUPlugin()
        plugin.optimizer_step(mock_optimizer, mock_model, mock_closure)

        mock_optimizer.step.assert_called_once()
        mock_closure.assert_not_called()

    @pytest.fixture()
    def mock_super(self, mocker) -> MagicMock:
        mock_super = MagicMock()
        mocker.patch.object(target_file, "super", return_value=mock_super)
        return mock_super

    @pytest.mark.parametrize("clip_val", [0, 1])
    def test_clip_gradients(
        self,
        plugin,
        mock_optimizer,
        clip_val,
        mock_super,
    ):
        plugin.clip_gradients(mock_optimizer, clip_val)
        mock_super.clip_gradients.assert_called_once()

    def test_clip_gradients_not_allowed_grad_clip(
        self,
        plugin,
        mock_optimizer,
    ):
        mock_optimizer._step_supports_amp_scaling = True
        with pytest.raises(RuntimeError, match="does not allow for gradient clipping"):
            plugin.clip_gradients(mock_optimizer, 0.1)

    @pytest.fixture()
    def mock_torch(self, mocker) -> MagicMock:
        return mocker.patch.object(target_file, "torch")

    def test_forward_context(self, plugin, mock_torch):
        with plugin.forward_context():
            pass

        mock_torch.xpu.autocast.assert_called_once_with(True)

    def test_state_dict(self, plugin, mock_scaler_state_dict):
        output = plugin.state_dict()
        assert output == mock_scaler_state_dict

    def test_state_dict_no_scaler(self):
        plugin = MixedPrecisionXPUPlugin()
        assert plugin.state_dict() == {}

    def test_load_state_dict(self, plugin, mock_scaler):
        mock_state_dict = MagicMock()
        plugin.load_state_dict(mock_state_dict)
        mock_scaler.load_state_dict.assert_called_once_with(mock_state_dict)
