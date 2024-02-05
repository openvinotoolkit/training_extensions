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
    def __init__(self, average: Literal["MICRO", "MACRO"]):
        self.average = average
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        pass
    
    def compute(self) -> Tensor:
        pass
    
    
    def _compute_unnormalized_confusion_metrices_from_preds(self, preds: Tensor, ):
        """Compute an (unnormalized) confusion matriix for every label group.""" 
        pass
    
    def _get_gt_and_pred_label_indices(self, preds: Tensor):
        pass