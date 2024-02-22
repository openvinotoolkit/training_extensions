# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection lightning module used in OTX."""
from __future__ import annotations

import inspect
import logging as log
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from otx.core.data.entity.detection import (
    DetBatchDataEntity,
    DetBatchPredEntity,
    DetBatchPredEntityWithXAI,
)
from otx.core.model.entity.detection import ExplainableOTXDetModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    from otx.algo.metrices import MetricCallable


class OTXDetectionLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX detection task."""

    def __init__(
        self,
        otx_model: ExplainableOTXDetModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable | None = None,
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
        )

        param_dict = {}
        if metric:
            sig = inspect.signature(metric)
            for name, param in sig.parameters.items():
                param_dict[name] = param.default
            param_dict.pop("kwargs", {})
            metric = metric(**param_dict)  # type: ignore[call-arg]

        self.metric = metric
        if self.metric is not None:
            self.metric.num_classes = otx_model.num_classes  # type: ignore[attr-defined]

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

    def validation_step(self, inputs: DetBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, (DetBatchPredEntity, DetBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if self.metric:
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: DetBatchPredEntity | DetBatchPredEntityWithXAI,
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

        if not isinstance(preds, (DetBatchPredEntity, DetBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if self.metric:
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )
