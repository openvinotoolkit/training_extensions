"""Anomaly XPU device callback."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from pytorch_lightning import Callback

from otx.algorithms.common.utils.utils import is_xpu_available


class XPUCallback(Callback):
    def __init__(self, device_idx=0):
        self.device = torch.device(f"xpu:{device_idx}")

    def on_fit_start(self, trainer, pl_module):
        if is_xpu_available():
            pl_module.to(self.device)
            model, optimizer = torch.xpu.optimize(trainer.model, optimizer=trainer.optimizers[0], dtype=torch.float32)
            trainer.optimizers = [optimizer]
            trainer.model = model

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        for k in batch:
            if not isinstance(batch[k], list):
                batch[k] = batch[k].to(self.device)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        for k in batch:
            batch[k] = batch[k].to(self.device)
