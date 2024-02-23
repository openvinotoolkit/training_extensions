# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for action detection lightning module used in OTX."""
from __future__ import annotations

import inspect
import logging as log
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from otx.core.data.entity.action_detection import (
    ActionDetBatchDataEntity,
    ActionDetBatchPredEntity,
)
from otx.core.model.entity.action_detection import OTXActionDetModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class OTXActionDetLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX detection task."""

    def __init__(
        self,
        otx_model: OTXActionDetModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = lambda: MeanAveragePrecision(),
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

    def configure_metric(self) -> None:
        """Configure the metric."""
        if isinstance(self.metric, partial):
            sig = inspect.signature(self.metric)
            param_dict = {}
            for name, param in sig.parameters.items():
                param_dict[name] = param.default
            param_dict.pop("kwargs", {})
            self.metric = self.metric(**param_dict)
        elif isinstance(self.metric, Metric):
            self.metric = self.metric

        if not isinstance(self.metric, Metric):
            msg = "Metric should be the instance of torchmetrics.Metric."
            raise TypeError(msg)

        # Since the metric is not initialized at the init phase,
        # Need to manually correct the device setting.
        self.metric.to(self.device)

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""
        self.configure_metric()

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
        self.configure_metric()

    def _log_metrics(self, meter: MeanAveragePrecision, key: str) -> None:
        results = meter.compute()
        if results is None:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

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
                sync_dist=True,
                prog_bar=True,
            )

    def validation_step(self, inputs: ActionDetBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)
        inputs.labels = [label.argmax(-1) for label in inputs.labels]

        if not isinstance(preds, ActionDetBatchPredEntity):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: ActionDetBatchPredEntity,
        inputs: ActionDetBatchDataEntity,
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

    def test_step(self, inputs: ActionDetBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)
        inputs.labels = [label.argmax(-1) for label in inputs.labels]

        if not isinstance(preds, ActionDetBatchPredEntity):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )
