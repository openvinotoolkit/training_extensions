# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any

import numpy as np
import torch
from torchmetrics import Metric, ConfusionMatrix

if TYPE_CHECKING:
    from otx.core.data.dataset.base import LabelInfo
    from torch import Tensor

class NamedConfusionMatrix:
    """Named Confusion Matrix to add row, col label names."""
    def __init__(
        self, 
        task: str, 
        num_classes: int,
        col_names: str,
        row_names: str,
    ):
        self.conf_matrix = ConfusionMatrix(task=task, num_classes=num_classes)
        self.col_names = col_names
        self.row_names = row_names
    
    def __call__(self, *args: Any, **kwargs: Any):
        return self.conf_matrix(*args, **kwargs)
    
    
class Accuracy(Metric):
    """Accuracy for the OTX classification tasks."""
    def __init__(self, task: Literal["multiclass", "multilabel", "hlabel"], 
                 average: Literal["MICRO", "MACRO"], label_info: LabelInfo):
        super().__init__()
        self.task = "multiclass"
        self.average = average
        self.label_groups = label_info.label_groups
        self.label_names = label_info.label_names
        
        self.preds = []
        self.targets = []

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.extend(preds)
        self.targets.extend(target)

    def _compute_preds_targets_for_multilabel(self, label_group: list[list[str]]) -> dict[str, torch.tensor]:
        targets = 
        
    def _compute_preds_targets_for_multiclass(self, label_group: list[list[str]]) -> dict[str, torch.tensor]:
        label_to_idx = {label: index for index, label in enumerate(self.label_names)}
        group_indices = [label_to_idx[label] for label in label_group]
        
        mask = torch.tensor([t.item() in group_indices for t in self.targets])
        filtered_preds = torch.tensor(self.preds)[mask]
        filtered_targets = torch.tensor(self.targets)[mask]

        for i, index in enumerate(group_indices):
            filtered_preds[filtered_preds == index] = i
            filtered_targets[filtered_targets == index] = i
        
        return {
            "preds": filtered_preds,
            "targets": filtered_targets
        }
    
    def _compute_unnormalized_confusion_matrics(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        conf_matrics = []
        for i, label_group in enumerate(self.label_groups):
            if len(label_group) == 1:
                compute_results = self._compute_preds_targets_for_multilabel(label_group)
            else:
                compute_results = self._compute_preds_targets_for_multiclass(label_group)
            num_classes = len(label_group)
            confmat = NamedConfusionMatrix(task=self.task, num_classes=num_classes, 
                                           row_names=label_group, col_names=label_group)
            conf_matrics.append(confmat(compute_results["preds"], compute_results["targets"]))
        return conf_matrics

    def _compute_accuracy_from_conf_matrics(self, conf_matrics: list[NamedConfusionMatrix]):
        """Compute the accuracy from the confusion matrix."""
        correct_per_label_group = [torch.diag(conf_matrix) for conf_matrix in conf_matrics]
        total_per_label_group = [torch.sum(conf_matrix) for conf_matrix in conf_matrics]
        
        if self.average == "MICRO":
            return np.sum(correct_per_label_group) / np.sum(total_per_label_group)
        elif self.average == "MACRO":
            return np.nanmean(np.divide(correct_per_label_group, total_per_label_group))
        else:
            msg = f"Average should be MICRO or MACRO, got {self.average}"
            raise ValueError(msg)

    def compute(self) -> Tensor:
        """Compute the metric."""
        conf_matrics = self._compute_unnormalized_confusion_matrics()
         
        return {
            "conf_matrix": conf_matrics,
            "accuracy": self._compute_accuracy_from_conf_matrics(conf_matrics) 
        }