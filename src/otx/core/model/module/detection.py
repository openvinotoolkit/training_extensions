# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection lightning module used in OTX."""
from __future__ import annotations

import logging as log

import torch
from torch import Tensor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from otx.core.data.entity.detection import (
    DetBatchDataEntity,
    DetBatchPredEntity,
)
from otx.core.model.entity.detection import OTXDetectionModel
from otx.core.model.module.base import OTXLitModule


class OTXDetectionLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX detection task."""

    def __init__(
        self,
        otx_model: OTXDetectionModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        torch_compile: bool,
    ):
        super().__init__(otx_model, optimizer, scheduler, torch_compile)

        self.val_metric = MeanAveragePrecision()
        self.test_metric = MeanAveragePrecision()

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.val_metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""
        self.test_metric.reset()

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        self._log_metrics(self.val_metric, "val")

    def on_test_epoch_end(self) -> None:
        """Callback triggered when the test epoch ends."""
        self._log_metrics(self.test_metric, "test")

    def _log_metrics(self, meter: MeanAveragePrecision, key: str) -> None:
        results = meter.compute()
        for k, v in results.items():
            if not isinstance(v, Tensor):
                log.debug("Cannot log item which is not Tensor")
                continue
            if v.numel() != 1:
                log.debug("Cannot log Tensor which is not scalar")
                continue

            self.log(
                f"{key}/{k}",
                v,
                sync_dist=False,
                prog_bar=True,
            )

    def validation_step(self, inputs: DetBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, DetBatchPredEntity):
            raise TypeError(preds)

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: DetBatchPredEntity,
        inputs: DetBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        return {
            "preds": [
                {
                    "boxes": bboxes.data,
                    "scores": scores,
                    "labels": labels,
                }
                for bboxes, scores, labels in zip(
                    preds.bboxes,
                    preds.scores,
                    preds.labels,
                )
            ],
            "target": [
                {
                    "boxes": bboxes.data,
                    "labels": labels,
                }
                for bboxes, labels in zip(inputs.bboxes, inputs.labels)
            ],
        }

    def test_step(self, inputs: DetBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, DetBatchPredEntity):
            raise TypeError(preds)

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"
