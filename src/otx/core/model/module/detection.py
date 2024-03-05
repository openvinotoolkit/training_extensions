# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection lightning module used in OTX."""
from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from otx.core.data.entity.detection import (
    DetBatchDataEntity,
    DetBatchPredEntity,
    DetBatchPredEntityWithXAI,
)
from otx.core.model.entity.detection import ExplainableOTXDetModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class OTXDetectionLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX detection task."""

    def __init__(
        self,
        otx_model: ExplainableOTXDetModel,
        torch_compile: bool,
        optimizer: list[OptimizerCallable] | OptimizerCallable = lambda p: torch.optim.SGD(p, lr=0.01),
        scheduler: list[LRSchedulerCallable] | LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR,
        metric: MetricCallable = lambda: MeanAveragePrecision(),
        warmup_steps: int = 0,
        warmup_by_epochs: bool = False
    ):
        super().__init__(
            otx_model=otx_model,
            torch_compile=torch_compile,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            warmup_steps=warmup_steps,
            warmup_by_epochs=warmup_by_epochs
        )
        self.test_meta_info: dict[str, Any] = self.model.test_meta_info if hasattr(self.model, "test_meta_info") else {}

    def load_state_dict(self, ckpt: dict[str, Any], *args, **kwargs) -> None:
        """Load state_dict from checkpoint.

        For detection, it is need to update confidence threshold information when
        the metric is FMeasure.
        """
        if "confidence_threshold" in ckpt:
            self.test_meta_info["best_confidence_threshold"] = ckpt["confidence_threshold"]
            self.test_meta_info["vary_confidence_threshold"] = False
        elif "confidence_threshold" in ckpt["hyper_parameters"]:
            self.test_meta_info["best_confidence_threshold"] = ckpt["hyper_parameters"]["confidence_threshold"]
            self.test_meta_info["vary_confidence_threshold"] = False
        super().load_state_dict(ckpt, *args, **kwargs)

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        for key, value in self.test_meta_info.items():
            if hasattr(self.metric, key):
                setattr(self.metric, key, value)

    def _log_metrics(self, meter: Metric, key: str) -> None:
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
        if hasattr(meter, "best_confidence_threshold"):
            self.hparams["confidence_threshold"] = meter.best_confidence_threshold

    def validation_step(self, inputs: DetBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, (DetBatchPredEntity, DetBatchPredEntityWithXAI)):
            raise TypeError(preds)

        if isinstance(self.metric, Metric):
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

        if isinstance(self.metric, Metric):
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )
