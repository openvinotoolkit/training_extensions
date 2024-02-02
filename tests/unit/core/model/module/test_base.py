# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for base model module."""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest
from lightning.pytorch.trainer import Trainer
from otx.algo.schedulers.warmup_schedulers import WarmupReduceLROnPlateau
from otx.core.model.entity.base import OTXModel
from otx.core.model.module.base import LinearWarmupScheduler, OTXLitModule
from torch.optim import Optimizer


class TestOTXLitModule:
    @pytest.fixture()
    def mock_otx_model(self) -> OTXModel:
        return create_autospec(OTXModel)

    @pytest.fixture()
    def mock_optimizer(self) -> Optimizer:
        optimizer = MagicMock(spec=Optimizer)
        optimizer.step = MagicMock()
        optimizer.keywords = {"lr": 0.01}
        optimizer.param_groups = MagicMock()

        def optimizer_factory(*args, **kargs) -> Optimizer:  # noqa: ARG001
            return optimizer

        return optimizer_factory

    @pytest.fixture()
    def mock_scheduler(self) -> WarmupReduceLROnPlateau:
        scheduler = MagicMock(spec=WarmupReduceLROnPlateau)
        scheduler.warmup_steps = 10

        def scheduler_factory(*args, **kargs) -> WarmupReduceLROnPlateau:  # noqa: ARG001
            return scheduler

        return scheduler_factory

    def test_configure_optimizers(self, mock_otx_model, mock_optimizer, mock_scheduler) -> None:
        module = OTXLitModule(
            otx_model=mock_otx_model,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
        )

        module.trainer = MagicMock(spec=Trainer)
        module.trainer.check_val_every_n_epoch = 2

        optimizers, lr_schedulers = module.configure_optimizers()
        assert isinstance(optimizers[0], Optimizer)
        assert isinstance(lr_schedulers[0]["scheduler"], LinearWarmupScheduler)
        assert lr_schedulers[0]["scheduler"].num_warmup_steps == 10
        assert lr_schedulers[0]["interval"] == "step"

        assert "scheduler" in lr_schedulers[1]
        assert "monitor" in lr_schedulers[1]
        assert "interval" in lr_schedulers[1]
        assert "frequency" in lr_schedulers[1]

        assert lr_schedulers[1]["frequency"] == 2
