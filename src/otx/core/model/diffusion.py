# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for diffusion model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.diffusion import DiffusionBatchDataEntity, DiffusionBatchPredEntity
from otx.core.metrics.diffusion import DiffusionMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters

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
        label_info: int = 0,
        **kwargs,
    ):
        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            **kwargs,
        )
        self.configure_metric()

    def _create_model(self) -> nn.Module:
        raise NotImplementedError

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()

        self.metric.persistent(True)
        self.metric.eval()

    def training_step(self, batch: DiffusionBatchDataEntity, batch_idx: int) -> torch.Tensor:
        """Step for model training."""
        train_loss = super().training_step(batch, batch_idx)
        if self.current_epoch == 0:
            self.metric.update(batch.images, real=True)
        return train_loss

    def _customize_outputs(
        self,
        outputs: torch.Tensor,
        inputs: DiffusionBatchDataEntity,
    ) -> DiffusionBatchPredEntity | OTXBatchLossEntity:
        return OTXBatchLossEntity(loss=outputs)

    def on_validation_start(self) -> None:
        """Called at the beginning of validation.

        Don't configure the metric here. Do it in constructor.
        """

    def on_test_start(self) -> None:
        """Called at the beginning of testing.

        Don't configure the metric here. Do it in constructor.
        """

    def test_step(self, batch: DiffusionBatchDataEntity, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.validation_step(batch, batch_idx)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(model_type="unet", task_type="diffusion")
