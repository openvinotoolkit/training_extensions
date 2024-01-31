# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for base model module."""

from __future__ import annotations

from unittest.mock import MagicMock, create_autospec

import pytest

from otx.core.model.module.base import OTXLitModule
from otx.core.model.entity.base import OTXModel
from otx.algo.schedulers.warmup_schedulers import WarmupReduceLROnPlateau

from torch.optim import Optimizer

from lightning.pytorch.trainer import Trainer

class TestOTXLitModule:
    @pytest.fixture()
    def mock_otx_model(self) -> OTXModel:
        return create_autospec(OTXModel)

    @pytest.fixture()
    def mock_optimizer(self) -> Optimizer:
        optimizer = MagicMock(spec=Optimizer)
        optimizer.keywords = {"lr": 0.01}
        return optimizer

    @pytest.fixture()
    def mock_scheduler(self) -> WarmupReduceLROnPlateau:
        scheduler = MagicMock(spec=WarmupReduceLROnPlateau)
        scheduler.warmup_steps = 10
        scheduler.warmup_by_epoch = True
        return scheduler

    def test_configure_optimizers(self, mock_otx_model, mock_optimizer, mock_scheduler):
        module = OTXLitModule(otx_model=mock_otx_model, torch_compile=False, optimizer=mock_optimizer, scheduler=mock_scheduler)
        module.trainer = MagicMock(spec=Trainer)
        module.trainer.check_val_every_n_epoch = 2
        
        optimizers = module.configure_optimizers()
        assert 'optimizer' in optimizers
        assert 'lr_scheduler' in optimizers
        assert 'monitor' in optimizers['lr_scheduler']
        assert 'interval' in optimizers['lr_scheduler']
        
        assert module.warmup_steps == 10
        assert module.warmup_by_epoch == True
        assert optimizers['lr_scheduler']['frequency'] == 2
    
    def test_optimizer_step_warmup_by_epoch(self, mock_otx_model, mock_optimizer, mock_scheduler):
        module = OTXLitModule(otx_model=mock_otx_model, torch_compile=False, optimizer=mock_optimizer, scheduler=mock_scheduler)
        module.learning_rate = 0.01
        module.warmup_steps = 10
        module.warmup_by_epoch = True
        module.trainer = MagicMock()
        module.trainer.current_epoch = 5

        param_group = {'lr': 0.01}
        mock_optimizer.param_groups = [param_group]

        module.optimizer_step(epoch=5, batch=0, optimizer=mock_optimizer, optimizer_closure=lambda: None)

        expected_lr = min(1.0, float(module.trainer.current_epoch + 1) / module.warmup_steps) * module.learning_rate
        assert mock_optimizer.param_groups[0]['lr'] == expected_lr

    def test_optimizer_step_warmup_by_steps(self, mock_otx_model, mock_optimizer, mock_scheduler):
        module = OTXLitModule(otx_model=mock_otx_model, torch_compile=False, optimizer=mock_optimizer, scheduler=mock_scheduler)
        module.learning_rate = 0.01
        module.warmup_steps = 10
        module.warmup_by_epoch = False
        module.trainer = MagicMock()
        module.trainer.global_step = 5

        param_group = {'lr': 0.01}
        mock_optimizer.param_groups = [param_group]

        module.optimizer_step(epoch=0, batch=5, optimizer=mock_optimizer, optimizer_closure=lambda: None)

        expected_lr = min(1.0, float(module.trainer.global_step + 1) / module.warmup_steps) * module.learning_rate
        assert mock_optimizer.param_groups[0]['lr'] == expected_lr