# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Tile Sampling Hook for OTXTileTrainDataset."""
from lightning import Callback, LightningModule, Trainer


class TileSamplingHook(Callback):
    """Hook to sample tiles from the training dataset."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Sample tiles from the training dataset when the epoch starts.

        Args:
            trainer (Trainer): PyTorch Lightning Trainer
            pl_module (LightningModule): Lightning Module

        """
        if (
            otx_train_ds := trainer.datamodule.subsets.get("train")
        ) is not None and otx_train_ds.tile_config.sampling_ratio < 1.0:
            new_subsample = otx_train_ds.sample_subset()
            otx_train_ds.dm_subset = new_subsample
            otx_train_ds.ids = [item.id for item in new_subsample]
        return super().on_train_epoch_start(trainer, pl_module)
