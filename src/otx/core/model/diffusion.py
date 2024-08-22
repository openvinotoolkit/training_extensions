# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for diffusion model entity used in OTX."""
from __future__ import annotations

from typing import TYPE_CHECKING

from otx.core.data.entity.diffusion import (
    DiffusionBatchDataEntity,
    DiffusionBatchPredEntity,
)
from otx.core.metrics.diffusion import DiffusionMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel
from otx.core.schedulers import LRSchedulerListCallable

if TYPE_CHECKING:
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

    def _create_model(self) -> nn.Module:
        raise NotImplementedError
