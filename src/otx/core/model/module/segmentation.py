# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for segmentation lightning module used in OTX."""
from __future__ import annotations

import logging as log

import torch
from torch import Tensor
from torchmetrics import JaccardIndex

from otx.core.config.export import ExportConfig
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
        export_config: ExportConfig,
    ):
        super().__init__(otx_model, optimizer, scheduler, torch_compile, export_config)
        num_classes = otx_model.config.get("decode_head", {}).get("num_classes", None)
        if num_classes is None:
            msg = """JaccardIndex metric cannot be used with num_classes = None.
            Please, specify number of classes in config."""
            raise RuntimeError(msg)

        metric_params = {
            "task": "multiclass",
            "num_classes": num_classes,
            "ignore_index": 255,
        }

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

        pred_list = self._convert_pred_entity_to_compute_metric(preds, inputs)
        for prediction in pred_list:
            self.test_metric.update(**prediction)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: SegBatchPredEntity,
        inputs: SegBatchDataEntity,
    ) -> list[dict[str, Tensor]]:
        return [
            {
                "preds": pred_mask,
                "target": target_mask,
            }
            for pred_mask, target_mask in zip(preds.masks, inputs.masks)
        ]

    def test_step(self, inputs: SegBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)
        if not isinstance(preds, SegBatchPredEntity):
            raise TypeError(preds)
        predictions = self._convert_pred_entity_to_compute_metric(preds, inputs)
        for prediction in predictions:
            self.test_metric.update(**prediction)

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"
