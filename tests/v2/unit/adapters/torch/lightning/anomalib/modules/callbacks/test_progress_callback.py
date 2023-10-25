"""Tests the progress callback on a dummy lightning module."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytorch_lightning as pl
from otx.v2.adapters.torch.lightning.anomalib.modules.callbacks import ProgressCallback
from otx.v2.api.entities.task_type import TaskType

from tests.v2.unit.adapters.torch.lightning.anomalib.helpers.dummy_dataset import DummyDataModule
from tests.v2.unit.adapters.torch.lightning.anomalib.helpers.dummy_model import DummyModel


class ProgressStageCheckerCallback(pl.Callback):
    """This callback is injected into the model to check the stage of progress callback.

    Args:
        progress_callback (ProgressCallback): Reference to progress callback.
    """

    def __init__(self, progress_callback: ProgressCallback) -> None:
        self.progress_callback = progress_callback

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        _, _ = trainer, pl_module
        assert self.progress_callback._get_progress("train") != 0.0

    def on_validation_batch_end(self, *args, **kwds) -> None:
        """Check that training progress is not 0.0 after validation batch end."""
        _, _ = args, kwds
        assert self.progress_callback._get_progress("train") != 0.0


class TestProgressCallback:
    def test_progress_callback(self) -> None:
        """Tests if progress callback runs and that the progress is not reset after validation step."""
        datamodule = DummyDataModule(TaskType.ANOMALY_CLASSIFICATION)
        model = DummyModel()
        progress_callback = ProgressCallback()
        # inject callback after progress callback
        stage_checker = ProgressStageCheckerCallback(progress_callback)
        # turn off sanity check on validation step as it will fail due to missing training data length
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            max_epochs=5,
            callbacks=[progress_callback, stage_checker],
            num_sanity_val_steps=0,
        )
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)
        trainer.predict(model, datamodule)
