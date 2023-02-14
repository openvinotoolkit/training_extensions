"""Dummy lightning modules for testing."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytorch_lightning as pl
import torch
from anomalib.data.utils import masks_to_boxes
from anomalib.utils.metrics import AnomalyScoreThreshold, MinMax


class DummyModel(pl.LightningModule):
    """Returns mock outputs."""

    def __init__(self):
        super().__init__()
        self.image_threshold = AnomalyScoreThreshold()
        self.normalization_metrics = MinMax()

    def configure_optimizers(self):
        return None

    def training_step(self, *args, **kwargs):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        # Just return everything as anomalous
        batch["anomaly_maps"] = batch["pred_masks"] = torch.ones(batch["image"].shape[0], 1, *batch["image"].shape[2:])
        batch["pred_labels"] = batch["pred_scores"] = torch.ones((batch["image"].shape[0]))
        batch["pred_boxes"], batch["box_scores"] = masks_to_boxes(batch["pred_masks"], batch["anomaly_maps"])
        is_anomalous = [scores > 0.5 for scores in batch["box_scores"]]
        batch["box_labels"] = [labels.int() for labels in is_anomalous]
        return batch

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self.predict_step(batch, batch_idx, dataloader_idx)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        return self.predict_step(batch, batch_idx, dataloader_idx)
