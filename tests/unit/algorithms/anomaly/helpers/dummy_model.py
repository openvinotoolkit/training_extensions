"""Dummy lightning modules for testing."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytorch_lightning as pl
import torch
from anomalib.utils.metrics import AdaptiveThreshold, MinMax


class DummyModel(pl.LightningModule):
    """Returns mock outputs."""

    def __init__(self):
        super().__init__()
        self.image_threshold = AdaptiveThreshold()
        self.normalization_metrics = MinMax()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # Just return everything as anomalous
        batch["anomaly_maps"] = batch["pred_masks"] = torch.ones(batch["image"].shape[0], 1, *batch["image"].shape[2:])
        batch["pred_labels"] = batch["pred_scores"] = torch.ones((batch["image"].shape[0]))

        return batch
