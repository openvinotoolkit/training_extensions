# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tests the XPU strategy."""


import pytest
import pytorch_lightning as pl
import torch
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from otx.algo.strategies import xpu_single as target_file
from otx.algo.strategies.xpu_single import SingleXPUStrategy


class TestSingleXPUStrategy:
    @pytest.fixture()
    def mock_is_xpu_available(self, mocker):
        return mocker.patch.object(target_file, "is_xpu_available", return_value=True)

    def test_init(self, mock_is_xpu_available):
        strategy = SingleXPUStrategy(device="xpu:0")
        assert mock_is_xpu_available.call_count == 1
        assert strategy._root_device.type == "xpu"
        assert strategy.accelerator is None

    def test_init_no_xpu(self, mock_is_xpu_available):
        mock_is_xpu_available.return_value = False
        with pytest.raises(MisconfigurationException):
            SingleXPUStrategy(device="xpu:0")

    @pytest.fixture()
    def strategy(self, mock_is_xpu_available):
        return SingleXPUStrategy(device="xpu:0", accelerator="xpu")

    def test_is_distributed(self, strategy):
        assert not strategy.is_distributed

    def test_setup_optimizers(self, strategy, mocker):
        from otx.algo.strategies.xpu_single import SingleDeviceStrategy

        mocker.patch("otx.algo.strategies.xpu_single.torch")
        mocker.patch(
            "otx.algo.strategies.xpu_single.torch.xpu.optimize",
            return_value=(mocker.MagicMock(), mocker.MagicMock()),
        )
        mocker.patch.object(SingleDeviceStrategy, "setup_optimizers")
        trainer = pl.Trainer()
        trainer.task = "CLASSIFICATION"
        # Create mock optimizers and models for testing
        model = torch.nn.Linear(10, 2)
        strategy._optimizers = [torch.optim.Adam(model.parameters(), lr=0.001)]
        strategy._model = model
        strategy.setup_optimizers(trainer)
        assert len(strategy.optimizers) == 1
