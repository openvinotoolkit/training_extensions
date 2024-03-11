# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from unittest.mock import MagicMock

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.cli import ReduceLROnPlateau
from lightning.pytorch.utilities.types import LRSchedulerConfig
from otx.algo.callbacks.adaptive_train_scheduling import AdaptiveTrainScheduling
from torch.utils.data import DataLoader


class TestAdaptiveTrainScheduling:
    def test_callback(self, caplog) -> None:
        callback = AdaptiveTrainScheduling(max_interval=5, decay=-0.025)

        mock_trainer = MagicMock(spec=Trainer)
        mock_pl_module = MagicMock(spec=LightningModule)

        mock_dataloader = MagicMock(spec=DataLoader)
        mock_dataloader.__len__.return_value = 32
        mock_trainer.train_dataloader = mock_dataloader

        mock_trainer.max_epochs = 10
        mock_trainer.check_val_every_n_epoch = 1
        mock_trainer.log_every_n_steps = 50

        mock_callback = MagicMock(spec=EarlyStopping)
        mock_callback.patience = 5
        mock_trainer.callbacks = [mock_callback]

        mock_lr_scheduler_config = MagicMock(spec=LRSchedulerConfig)
        mock_lr_scheduler_config.scheduler = MagicMock(spec=ReduceLROnPlateau)
        mock_lr_scheduler_config.scheduler.patience = 4
        mock_lr_scheduler_config.frequency = 1
        mock_lr_scheduler_config.interval = "epoch"
        mock_trainer.lr_scheduler_configs = [mock_lr_scheduler_config]

        with caplog.at_level(log.WARNING):
            callback.on_train_start(trainer=mock_trainer, pl_module=mock_pl_module)
            assert mock_trainer.check_val_every_n_epoch != 1  # Adaptively updated, in this case, 2
            assert mock_trainer.callbacks[0].patience != 5
            assert mock_trainer.lr_scheduler_configs[0].frequency != 1
            assert mock_trainer.lr_scheduler_configs[0].scheduler.patience == 1  # int((4+1) / 2) - 1 = 1
            assert mock_trainer.log_every_n_steps == 32  # Equal to len(train_dataloader)
            assert len(caplog.records) == 5  # Warning two times

        callback.on_train_end(trainer=mock_trainer, pl_module=mock_pl_module)

        # Restore temporarily updated values
        assert mock_trainer.check_val_every_n_epoch == 1
        assert mock_trainer.log_every_n_steps == 50
        assert mock_trainer.callbacks[0].patience == 5
        assert mock_trainer.lr_scheduler_configs[0].frequency == 1
        assert mock_trainer.lr_scheduler_configs[0].scheduler.patience == 4
