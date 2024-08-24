# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for diffusion model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.data.entity.diffusion import DiffusionBatchDataEntity, DiffusionBatchPredEntity
from otx.core.metrics.diffusion import DiffusionMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel
from otx.core.schedulers import LRSchedulerListCallable

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.core.metrics import MetricCallable


class OTXDiffusionModel(OTXModel[DiffusionBatchDataEntity, DiffusionBatchPredEntity]):
    """OTX Diffusion model."""

    def __init__(
        self,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: (LRSchedulerCallable | LRSchedulerListCallable) = DefaultSchedulerCallable,
        metric: MetricCallable = DiffusionMetricCallable,
        torch_compile: bool = False,
    ):
        super().__init__(
            label_info=0,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.epoch_idx = 0

    def _create_model(self) -> nn.Module:
        raise NotImplementedError

    def on_fit_start(self) -> None:
        """Called at the very beginning of fit.

        If on DDP it is called on every process

        """
        self.configure_metric()

    def training_step(self, batch: DiffusionBatchDataEntity, batch_idx: int) -> torch.Tensor:
        """Step for model training."""
        train_loss = super().training_step(batch, batch_idx)
        if self.epoch_idx == 0:
            self.metric.update(batch.images, real=True)
        return train_loss

    def test_step(self, batch: DiffusionBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.validation_step(batch, batch_idx)

    def on_validation_epoch_start(self) -> None:
        """Callback triggered when the validation epoch starts."""
        self.epoch_idx += 1

    def on_validation_epoch_end(self) -> None:
        """Callback triggered when the validation epoch ends."""
        super().on_validation_epoch_end()
        self.metric.reset()

    def on_test_epoch_start(self) -> None:
        """Callback triggered when the test epoch starts."""

    def on_validation_start(self) -> None:
        """Called at the beginning of validation."""

    def on_test_start(self) -> None:
        """Called at the beginning of testing."""
