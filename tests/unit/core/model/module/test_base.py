# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for base model module."""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest
from lightning.pytorch.cli import ReduceLROnPlateau
from lightning.pytorch.trainer import Trainer
from otx.core.model.entity.base import OTXModel
from otx.core.model.module.base import OTXLitModule
from torch.optim import Optimizer


class TestOTXLitModule:
    @pytest.fixture()
    def mock_otx_model(self) -> OTXModel:
        return create_autospec(OTXModel)

    @pytest.fixture()
    def mock_optimizer(self) -> list[Optimizer]:
        optimizer = MagicMock(spec=Optimizer)
        optimizer.step = MagicMock()
        optimizer.keywords = {"lr": 0.01}
        optimizer.param_groups = MagicMock()

        def optimizer_factory(*args, **kargs) -> Optimizer:  # noqa: ARG001
            return optimizer

        return [optimizer_factory]

    @pytest.fixture()
    def mock_scheduler(self) -> list[ReduceLROnPlateau]:
        scheduler_object = MagicMock()

        lr_scheduler = MagicMock(spec=ReduceLROnPlateau)
        lr_scheduler.monitor = "val/loss"
        scheduler_object.return_value = lr_scheduler

        return [scheduler_object]

    def test_configure_optimizers(self, mock_otx_model, mock_optimizer, mock_scheduler) -> None:
        module = OTXLitModule(
            otx_model=mock_otx_model,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            metric=MagicMock(),
        )

        module.trainer = MagicMock(spec=Trainer)
        module.trainer.check_val_every_n_epoch = 2

        optimizers, lr_schedulers = module.configure_optimizers()
        assert isinstance(optimizers[0], Optimizer)

        assert "scheduler" in lr_schedulers[0]
        assert "monitor" in lr_schedulers[0]
        assert module.warmup_steps == 0
        assert module.warmup_by_epoch is False

    def test_optimzier_step_by_iter(self, mock_otx_model, mock_optimizer, mock_scheduler) -> None:
        module = OTXLitModule(
            otx_model=mock_otx_model,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            metric=MagicMock(),
            warmup_steps=10,
            warmup_by_epochs=False,
        )
        optimizers, _ = module.configure_optimizers()
        module.init_lr = 0.01

        module.trainer = MagicMock()
        module.trainer.global_step = 5

        param_group = {"lr": 0.01}
        optimizers[0].param_groups = [param_group]

        module.optimizer_step(epoch=0, batch=5, optimizer=optimizers[0], closure=lambda: None)
        expected_lr = min(1.0, float(module.trainer.global_step + 1) / module.warmup_steps) * module.init_lr
        assert optimizers[0].param_groups[0]["lr"] == expected_lr

    def test_optimzier_step_by_epoch(self, mock_otx_model, mock_optimizer, mock_scheduler) -> None:
        module = OTXLitModule(
            otx_model=mock_otx_model,
            torch_compile=False,
            optimizer=mock_optimizer,
            scheduler=mock_scheduler,
            metric=MagicMock(),
            warmup_steps=10,
            warmup_by_epochs=True,
        )
        optimizers, _ = module.configure_optimizers()
        module.init_lr = 0.01

        module.trainer = MagicMock()
        module.trainer.current_epoch = 5

        param_group = {"lr": 0.01}
        optimizers[0].param_groups = [param_group]

        module.optimizer_step(epoch=5, batch=0, optimizer=optimizers[0], closure=lambda: None)
        expected_lr = min(1.0, float(module.trainer.current_epoch + 1) / module.warmup_steps) * module.init_lr
        assert optimizers[0].param_groups[0]["lr"] == expected_lr
