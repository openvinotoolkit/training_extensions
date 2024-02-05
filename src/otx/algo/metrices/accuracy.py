# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torchmetrics import Metric

if TYPE_CHECKING:
    from torch import Tensor


class Accuracy(Metric):
    """Skleton code for future implementation."""

    def __init__(self, average: Literal["MICRO", "MACRO"]):
        self.average = average

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""

    def compute(self) -> Tensor:
        """Compute the metric."""

    def _compute_unnormalized_confusion_metrices_from_preds(self, preds: Tensor) -> None:
        """Compute an (unnormalized) confusion matriix for every label group."""
