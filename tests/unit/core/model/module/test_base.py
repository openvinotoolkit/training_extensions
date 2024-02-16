# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for base model module."""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from lightning.pytorch.trainer import Trainer
from otx.algo.schedulers.warmup_schedulers import LinearWarmupScheduler
from otx.core.model.entity.base import OTXModel
from otx.core.model.module.base import OTXLitModule
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
    def mock_scheduler(self) -> list[LinearWarmupScheduler | ReduceLROnPlateau]:
        scheduler_object_1 = MagicMock()
        warmup_scheduler = MagicMock(spec=LinearWarmupScheduler)
        warmup_scheduler.num_warmup_steps = 10
        warmup_scheduler.interval = "step"
        scheduler_object_1.return_value = warmup_scheduler

        scheduler_object_2 = MagicMock()
        lr_scheduler = MagicMock(spec=ReduceLROnPlateau)
        lr_scheduler.monitor = "val/loss"
        scheduler_object_2.return_value = lr_scheduler

        return [scheduler_object_1, scheduler_object_2]

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
