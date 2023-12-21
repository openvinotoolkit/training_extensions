# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for classification lightning module used in OTX."""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional
from torchmetrics.classification import MultilabelAccuracy
from torchmetrics.classification.accuracy import Accuracy

from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.model.entity.classification import OTXMulticlassClsModel, OTXMultilabelClsModel
from otx.core.model.module.base import OTXLitModule


class OTXMulticlassClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX multi-class classification task."""

    def __init__(
        self,
        otx_model: OTXMulticlassClsModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        torch_compile: bool,
    ):
        super().__init__(otx_model, optimizer, scheduler, torch_compile)
        num_classes = otx_model.config.get("head", {}).get("num_classes", None)
        self.val_metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_metric = Accuracy(task="multiclass", num_classes=num_classes)

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

    def _log_metrics(self, meter: Accuracy, key: str) -> None:
        results = meter.compute()
        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: MulticlassClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, MulticlassClsBatchPredEntity):
            raise TypeError(preds)

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MulticlassClsBatchPredEntity,
        inputs: MulticlassClsBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }

    def test_step(self, inputs: MulticlassClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, MulticlassClsBatchPredEntity):
            raise TypeError(preds)

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"


class OTXMultilabelClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX multi-label classification task."""

    def __init__(
        self,
        otx_model: OTXMultilabelClsModel,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        torch_compile: bool,
    ):
        super().__init__(otx_model, optimizer, scheduler, torch_compile)
        self.num_labels = otx_model.config.get("head", {}).get("num_classes", None)

        self.val_metric = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5, average="micro")
        self.test_metric = MultilabelAccuracy(num_labels=self.num_labels, threshold=0.5, average="micro")

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

    def _log_metrics(self, meter: Accuracy, key: str) -> None:
        results = meter.compute()
        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: MultilabelClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, MultilabelClsBatchPredEntity):
            raise TypeError(preds)

        self.val_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MultilabelClsBatchPredEntity,
        inputs: MultilabelClsBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        return {
            "preds": torch.stack(preds.scores),
            "target": torch.stack(inputs.labels),
        }

    def test_step(self, inputs: MultilabelClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, MultilabelClsBatchPredEntity):
            raise TypeError(preds)

        self.test_metric.update(
            **self._convert_pred_entity_to_compute_metric(preds, inputs),
        )

    @property
    def lr_scheduler_monitor_key(self) -> str:
        """Metric name that the learning rate scheduler monitor."""
        return "train/loss"
