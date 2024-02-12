# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
""" Tile Sampling Hook for  OTXTileTrainDataset""" 
from lightning import Callback, LightningModule, Trainer


class TileSamplingHook(Callback):
    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if(otx_train_ds := trainer.datamodule.subsets.get("train")) is not None:
            new_subsample = otx_train_ds.sample_subset()
            otx_train_ds.dm_subset = new_subsample
            otx_train_ds.ids = [item.id for item in new_subsample]
        return super().on_train_epoch_start(trainer, pl_module)