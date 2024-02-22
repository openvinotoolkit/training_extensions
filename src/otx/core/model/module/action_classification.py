# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for action classification lightning module used in OTX."""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from otx.core.data.entity.action_classification import (
    ActionClsBatchDataEntity,
    ActionClsBatchPredEntity,
)
from otx.core.model.entity.action_classification import OTXActionClsModel
from otx.core.model.module.base import OTXLitModule

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torchmetrics.classification.accuracy import Accuracy

    from otx.algo.metrices import MetricCallable


class OTXActionClsLitModule(OTXLitModule):
    """Base class for the lightning module used in OTX detection task."""

    def __init__(
        self,
        otx_model: OTXActionClsModel,
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

        if metric:
            if inspect.isclass(metric):
                sig = inspect.signature(metric)
                param_dict = {}
                for name, param in sig.parameters.items():
                    param_dict[name] = param.default if name != "num_classes" else self.model.num_classes
                param_dict.pop("kwargs", {})
                metric = metric(**param_dict)
            else:
                msg = "Function based metric not yet supported."
                raise ValueError(msg)

        self.metric = metric

    def _log_metrics(self, meter: Accuracy, key: str) -> None:
        results = meter.compute()
        if results is None:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)

        self.log(f"{key}/accuracy", results.item(), sync_dist=True, prog_bar=True)

    def validation_step(self, inputs: ActionClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, ActionClsBatchPredEntity):
            raise TypeError(preds)

        if self.metric:
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: ActionClsBatchPredEntity,
        inputs: ActionClsBatchDataEntity,
    ) -> dict[str, list[dict[str, Tensor]]]:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }

    def test_step(self, inputs: ActionClsBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.model(inputs)

        if not isinstance(preds, ActionClsBatchPredEntity):
            raise TypeError(preds)

        if self.metric:
            self.metric.update(
                **self._convert_pred_entity_to_compute_metric(preds, inputs),
            )
