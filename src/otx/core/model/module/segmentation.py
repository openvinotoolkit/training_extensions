# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for segmentation lightning module used in OTX."""
from __future__ import annotations

import inspect
import logging as log
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torchmetrics import Dice, Metric

from otx.core.data.entity.segmentation import (
    SegBatchDataEntity,
    SegBatchPredEntity,
    SegBatchPredEntityWithXAI,
)
from otx.core.model.entity.segmentation import OTXSegmentationModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrices import MetricCallable


class OTXSegmentationLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX segmentation task."""

    def __init__(
        self,
        otx_model: OTXSegmentationModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = lambda: Dice(),
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
        if isinstance(self.metric_callable, partial):
            sig = inspect.signature(self.metric_callable)
            param_dict = {}
            for name, param in sig.parameters.items():
                if name == "num_classes":
                    param_dict[name] = self.model.num_classes + 1
                    param_dict["ignore_index"] = self.model.num_classes
                else:
                    param_dict[name] = param.default
            param_dict.pop("kwargs", {})
            self.metric = self.metric_callable(**param_dict)
        elif isinstance(self.metric_callable, Metric):
            self.metric = self.metric_callable

        if not isinstance(self.metric, Metric):
            msg = "Metric should be the instance of torchmetrics.Metric."
            raise TypeError(msg)

        # Since the metric is not initialized at the init phase,
        # Need to manually correct the device setting.
        self.metric.to(self.device)

    def _log_metrics(self, meter: Dice, key: str) -> None:
        results = meter.compute()
        if results is None:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

        if isinstance(results, Tensor):
            if results.numel() != 1:
                log.debug("Cannot log Tensor which is not scalar")
                return
            self.log(
                f"{key}/{type(meter).__name__}",
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

        if not isinstance(preds, (SegBatchPredEntity, SegBatchPredEntityWithXAI)):
            raise TypeError(preds)

        predictions = self._convert_pred_entity_to_compute_metric(preds, inputs)
        if isinstance(self.metric, Metric):
            for prediction in predictions:
                self.metric.update(**prediction)

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: SegBatchPredEntity | SegBatchPredEntityWithXAI,
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
        if not isinstance(preds, (SegBatchPredEntity, SegBatchPredEntityWithXAI)):
            raise TypeError(preds)
        predictions = self._convert_pred_entity_to_compute_metric(preds, inputs)
        if isinstance(self.metric, Metric):
            for prediction in predictions:
                self.metric.update(**prediction)
