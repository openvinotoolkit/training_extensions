# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Any

import torch
from torchmetrics import Metric, ConfusionMatrix

if TYPE_CHECKING:
    from otx.core.data.dataset.base import LabelInfo
    from torch import Tensor

class NamedConfusionMatrix(ConfusionMatrix):
    """Named Confusion Matrix to add row, col label names."""
    def __init__(
        self, 
        task: str, 
        num_classes: int, 
        row_names: list[str], 
        col_names: list[str]
    ):
        super().__init__(task, num_classes)
        self.row_names = row_names
        self.col_names = col_names
    
    def forward(self, *args: Any, **kwargs: Any):
        res = super().forward(*args, **kwargs)
        return {
            "matrix_values": res,
            "row_names": self.row_names,
            "col_names": self.col_names
        }
        

class Accuracy(Metric):
    """Accuracy for the OTX classification tasks."""

    def __init__(self, average: Literal["MICRO", "MACRO"], meta_info: LabelInfo):
        super().__init__()
        self.average = average
        self.label_groups = meta_info.label_groups
        self.label_names = meta_info.label_names
        
        self.preds = []
        self.targets = []

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.extend(preds)
        self.targets.extend(target)

    def _compute_unnormalized_confusion_matrix(self) -> list[NamedConfusionMatrix]:
        """Compute an (unnormalized) confusion matrix for every label group."""
        conf_matrics = []
        for i, label_group in self.label_groups:
            label_to_idx = {label: index for index, label in enumerate(self.label_names)}
            group_indices = [label_to_idx[label] for label in label_group]
            
            mask = torch.tensor([t.item() in group_indices for t in self.targets])
            filtered_preds = self.preds[mask]
            filtered_targets = self.targets[mask]
            

            for i, index in enumerate(group_indices):
                filtered_preds[filtered_preds == index] = i
                filtered_targets[filtered_targets == index] = i
            
            num_classes = len(label_group)
            confmat = NamedConfusionMatrix(task="multiclass", num_classes=num_classes,
                                        row_names=label_group, col_names=label_group)
            conf_matrics.append(confmat(filtered_preds, filtered_targets))
        return conf_matrics

    def _compute_accuracy_from_conf_matrix(self, conf_matrix: NamedConfusionMatrix):
        """Compute the accuracy from the confusion matrix."""
        true_positives = torch.diag(conf_matrix).sum().item()
        total = conf_matrix.sum().item()
        return true_positives / total if total > 0 else 0

    def compute(self) -> Tensor:
        """Compute the metric."""
        conf_matrix = self._compute_accuracy_from_conf_matrix()
        return {
            "conf_matrix": conf_matrix,
            "accuracy": self._compute_accuracy_from_conf_matrix(conf_matrix)
        }
        

class MulticlassAccuracy(Accuracy):
    """Skleton code for future implementation."""
    def compute(self) -> Tensor:
        """Compute the metric."""
        conf_matrics = self._compute_unnormalized_confusion_matrix()

        
