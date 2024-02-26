# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torchmetrics import ConfusionMatrix, Metric

if TYPE_CHECKING:
    from torch import Tensor

    from otx.core.data.dataset.base import LabelInfo


class NamedConfusionMatrix(nn.Module):
    """Named Confusion Matrix to add row, col label names."""

    def __init__(
        self,
        task: str,
        num_classes: int,
        col_names: list[str],
        row_names: list[str],
    ):
        super().__init__()
        self.conf_matrix = ConfusionMatrix(task=task, num_classes=num_classes)
        self.col_names = col_names
        self.row_names = row_names

    def __call__(self, *args: object, **kwargs: object):
        """Call function of the Named Confusion Matrix."""
        return self.conf_matrix(*args, **kwargs)


class CustomAccuracy(Metric):
    """Accuracy for the OTX classification tasks."""

    def __init__(self, label_info: LabelInfo, average: Literal["MICRO", "MACRO"] = "MICRO", threshold: float = 0.5):
        super().__init__()
        self.average = average
        self.label_info = label_info
        self.label_groups = label_info.label_groups
        self.label_names = label_info.label_names

        self.threshold = threshold

        self.preds: list[Tensor] = []
        self.targets: list[Tensor] = []

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.extend(preds)
        self.targets.extend(target)

    def _compute_unnormalized_confusion_matrics(self) -> list[NamedConfusionMatrix]:
        raise NotImplementedError

    def _compute_accuracy_from_conf_matrics(self, conf_matrics: list[NamedConfusionMatrix]) -> Tensor:
        """Compute the accuracy from the confusion matrix."""
        correct_per_label_group = torch.stack([torch.trace(conf_matrix) for conf_matrix in conf_matrics])
        total_per_label_group = torch.stack([torch.sum(conf_matrix) for conf_matrix in conf_matrics])

        if self.average == "MICRO":
            return torch.sum(correct_per_label_group) / torch.sum(total_per_label_group)
        if self.average == "MACRO":
            return torch.nanmean(torch.divide(correct_per_label_group, total_per_label_group))

        msg = f"Average should be MICRO or MACRO, got {self.average}"
        raise ValueError(msg)

    def compute(self) -> Tensor:
        """Compute the metric."""
        conf_matrics = self._compute_unnormalized_confusion_matrics()

        return {
            "conf_matrix": conf_matrics,
            "accuracy": self._compute_accuracy_from_conf_matrics(conf_matrics),
        }


class CustomMulticlassAccuracy(CustomAccuracy):
    """Custom accuracy class for the multi-class classification."""

    def _compute_unnormalized_confusion_matrics(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        conf_matrics = []
        for label_group in self.label_groups:
            label_to_idx = {label: index for index, label in enumerate(self.label_names)}
            group_indices = [label_to_idx[label] for label in label_group]

            mask = torch.tensor([t.item() in group_indices for t in self.targets])
            filtered_preds = torch.tensor(self.preds)[mask]
            filtered_targets = torch.tensor(self.targets)[mask]

            for i, index in enumerate(group_indices):
                filtered_preds[filtered_preds == index] = i
                filtered_targets[filtered_targets == index] = i

            num_classes = len(label_group)
            confmat = NamedConfusionMatrix(
                task="multiclass",
                num_classes=num_classes,
                row_names=label_group,
                col_names=label_group,
            ).to(self.device)
            conf_matrics.append(confmat(filtered_preds, filtered_targets))
        return conf_matrics


class CustomMultilabelAccuracy(CustomAccuracy):
    """Custom accuracy class for the multi-label classification."""

    def _compute_unnormalized_confusion_matrics(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        preds = torch.stack(self.preds)
        targets = torch.stack(self.targets)

        conf_matrics = []
        for i, label_group in enumerate(self.label_groups):
            label_preds = (preds[:, i] >= self.threshold).long()
            label_targets = targets[:, i]

            valid_mask = label_targets >= 0
            if valid_mask.any():
                valid_preds = label_preds[valid_mask]
                valid_targets = label_targets[valid_mask]
            else:
                continue

            data_name = [label_group[0], "~" + label_group[0]]
            confmat = NamedConfusionMatrix(task="binary", num_classes=2, row_names=data_name, col_names=data_name).to(
                self.device,
            )
            conf_matrics.append(confmat(valid_preds, valid_targets))
        return conf_matrics


class CustomHlabelAccuracy(CustomAccuracy):
    """Custom accuracy class for the hierarchical-label classification."""

    def _is_multiclass_group(self, label_group: list[str]) -> bool:
        return len(label_group) != 1

    def _compute_unnormalized_confusion_matrics(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        preds = torch.stack(self.preds)
        targets = torch.stack(self.targets)

        conf_matrics = []
        for i, label_group in enumerate(self.label_groups):
            label_preds = preds[:, i]
            label_targets = targets[:, i]

            valid_mask = label_targets >= 0
            if valid_mask.any():
                valid_preds = label_preds[valid_mask]
                valid_targets = label_targets[valid_mask]
            else:
                continue

            if self._is_multiclass_group(label_group):
                num_classes = len(label_group)
                confmat = NamedConfusionMatrix(
                    task="multiclass",
                    num_classes=num_classes,
                    row_names=label_group,
                    col_names=label_group,
                ).to(self.device)
                conf_matrics.append(confmat(valid_preds, valid_targets))
            else:
                label_preds = (label_preds >= self.threshold).long()
                data_name = [label_group[0], "~" + label_group[0]]
                confmat = NamedConfusionMatrix(
                    task="binary",
                    num_classes=2,
                    row_names=data_name,
                    col_names=data_name,
                ).to(self.device)
                conf_matrics.append(confmat(valid_preds, valid_targets))
        return conf_matrics
