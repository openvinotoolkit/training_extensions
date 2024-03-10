"""Tests the XPU strategy."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import pytorch_lightning as pl
from otx.algorithms.anomaly.adapters.anomalib.strategies.xpu_single import SingleXPUStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class TestSingleXPUStrategy:
    def test_init(self, mocker):
        with pytest.raises(MisconfigurationException):
            strategy = SingleXPUStrategy(device="xpu:0")
        mocked_is_xpu_available = mocker.patch(
            "otx.algorithms.anomaly.adapters.anomalib.strategies.xpu_single.is_xpu_available", return_value=True
        )
        strategy = SingleXPUStrategy(device="xpu:0")
        assert mocked_is_xpu_available.call_count == 1
        assert strategy._root_device.type == "xpu"
        assert strategy.accelerator is None

    @pytest.fixture
    def strategy(self, mocker):
        mocker.patch(
            "otx.algorithms.anomaly.adapters.anomalib.strategies.xpu_single.is_xpu_available", return_value=True
        )
        return SingleXPUStrategy(device="xpu:0")

    def test_is_distributed(self, strategy):
        assert not strategy.is_distributed

    def test_setup_optimizers(self, strategy, mocker):
        mocker.patch("otx.algorithms.anomaly.adapters.anomalib.strategies.xpu_single.torch")
        mocker.patch(
            "otx.algorithms.anomaly.adapters.anomalib.strategies.xpu_single.torch.xpu.optimize",
            return_value=(mocker.MagicMock(), mocker.MagicMock()),
        )
        trainer = pl.Trainer()
        # Create mock optimizers and models for testing
        model = torch.nn.Linear(10, 2)
        strategy._optimizers = [torch.optim.Adam(model.parameters(), lr=0.001)]
        strategy._model = model
        trainer.model = model
        strategy.setup_optimizers(trainer)
        assert len(strategy.optimizers) == 1
