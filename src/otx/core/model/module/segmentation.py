# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for segmentation lightning module used in OTX."""
from __future__ import annotations

import logging as log

import torch
from torch import Tensor
from torchmetrics import JaccardIndex

from otx.core.data.entity.segmentation import (
    SegBatchDataEntity,
    SegBatchPredEntity,
)
from otx.core.model.entity.segmentation import OTXSegmentationModel
from otx.core.model.module.base import OTXLitModule


class OTXSegmentationLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX segmentation task."""

    def __init__(
        self,
        otx_model: OTXSegmentationModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        torch_compile: bool,
    ):
        super().__init__(otx_model, optimizer, scheduler, torch_compile)
        metric_params = {"task": "multiclass",
                         "num_classes": otx_model.model.decode_head.num_classes,
                         "ignore_index": 255}

        self.val_metric = JaccardIndex(**metric_params)
        self.test_metric = JaccardIndex(**metric_params)

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

    def _log_metrics(self, meter: JaccardIndex, key: str) -> None:
        results = meter.compute()

        if isinstance(results, Tensor):
            if results.numel() != 1:
                log.debug("Cannot log Tensor which is not scalar")
                return
            self.log(
                f"{key}/mIoU",
                results,
                sync_dist=True,
                prog_bar=True,
            )
        else:
            log.debug("Cannot log item which is not Tensor")


    def validation_step(self, inputs: SegBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, SegBatchPredEntity):
            raise TypeError(preds)

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: SegBatchPredEntity,
        inputs: SegBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:

        return {
            "preds": torch.cat(preds.masks, dim=0),
            "target": torch.cat(inputs.masks, dim=0),
        }

    def test_step(self, inputs: SegBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)
        if not isinstance(preds, SegBatchPredEntity):
            raise TypeError(preds)

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"
