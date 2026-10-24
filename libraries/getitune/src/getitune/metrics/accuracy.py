# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for getitune accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor
from torchmetrics import ConfusionMatrix, Metric
from torchmetrics.classification import MulticlassF1Score as TorchmetricMulticlassF1
from torchmetrics.classification.accuracy import Accuracy as TorchmetricAcc
from torchmetrics.collections import MetricCollection

from getitune.metrics.types import MetricCallable

from .mlc_map import MultilabelmAP

if TYPE_CHECKING:
    from getitune.types.label import LabelInfo


class NamedConfusionMatrix(ConfusionMatrix):
    """Confusion matrix with row and column label names."""

    def __new__(
        cls,
        col_names: list[str],
        row_names: list[str],
        task: Literal["binary", "multiclass", "multilabel"],
        threshold: float = 0.5,
        num_classes: int | None = None,
        num_labels: int | None = None,
        normalize: Literal["true", "pred", "all", "none"] | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: object,
    ) -> NamedConfusionMatrix:
        """Create a confusion matrix and attach its row and column names."""
        confusion_metric = super().__new__(
            cls,
            task=task,
            threshold=threshold,
            num_classes=num_classes,
            num_labels=num_labels,
            normalize=normalize,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs,
        )
        confusion_metric.col_names = col_names
        confusion_metric.row_names = row_names
        return confusion_metric


class AccuracywithLabelGroup(Metric):
    """Accuracy calculated per label group."""

    def __init__(self, label_info: LabelInfo, *, average: Literal["MICRO", "MACRO"] = "MICRO", threshold: float = 0.5):
        super().__init__()
        self.average = average
        self.threshold = threshold
        self._label_info: LabelInfo = label_info
        self.preds: list[Tensor] = []
        self.targets: list[Tensor] = []

    @property
    def label_info(self) -> LabelInfo:
        """Return the label information used by the metric."""
        return self._label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfo) -> None:
        """Update the label information used by the metric."""
        self._label_info = label_info

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Accumulate predictions and targets for a batch."""
        self.preds.extend(preds)
        self.targets.extend(target)

    def _compute_unnormalized_confusion_matrices(self) -> list[Tensor]:
        raise NotImplementedError

    def _compute_accuracy_from_conf_matrices(self, conf_matrices: list[Tensor]) -> Tensor:
        correct = torch.stack([torch.trace(conf_matrix) for conf_matrix in conf_matrices])
        total = torch.stack([torch.sum(conf_matrix) for conf_matrix in conf_matrices])
        if self.average == "MICRO":
            return torch.sum(correct) / torch.sum(total)
        if self.average == "MACRO":
            return torch.nanmean(torch.divide(correct, total))
        msg = f"Average should be MICRO or MACRO, got {self.average}"
        raise ValueError(msg)

    def compute(self) -> dict[str, Any]:
        """Compute confusion matrices and aggregate accuracy."""
        conf_matrices = self._compute_unnormalized_confusion_matrices()
        return {"conf_matrix": conf_matrices, "accuracy": self._compute_accuracy_from_conf_matrices(conf_matrices)}


class MulticlassAccuracywithLabelGroup(AccuracywithLabelGroup):
    """Accuracy for multi-class classification with label groups."""

    def _compute_unnormalized_confusion_matrices(self) -> list[Tensor]:
        conf_matrices = []
        for label_group in self.label_info.label_groups:
            label_to_idx = {label: index for index, label in enumerate(self.label_info.label_names)}
            group_indices = [label_to_idx[label] for label in label_group]
            mask = torch.tensor([t.item() in group_indices for t in self.targets])
            valid_preds = torch.tensor(self.preds)[mask]
            valid_targets = torch.tensor(self.targets)[mask]
            for i, index in enumerate(group_indices):
                valid_preds[valid_preds == index] = i
                valid_targets[valid_targets == index] = i
            confmat = NamedConfusionMatrix(
                task="multiclass",
                num_classes=len(label_group),
                row_names=label_group,
                col_names=label_group,
            )
            conf_matrices.append(confmat(valid_preds, valid_targets))
        return conf_matrices


class MultilabelAccuracywithLabelGroup(AccuracywithLabelGroup):
    """Accuracy for multi-label classification with label groups."""

    def _compute_unnormalized_confusion_matrices(self) -> list[Tensor]:
        preds = torch.stack(self.preds)
        targets = torch.stack(self.targets)
        conf_matrices = []
        for i, label_name in enumerate(self.label_info.label_names):
            label_preds = (preds[:, i] >= self.threshold).long()
            label_targets = targets[:, i]
            valid_mask = label_targets >= 0
            if not valid_mask.any():
                continue
            data_name = [label_name, "~" + label_name]
            confmat = NamedConfusionMatrix(
                task="binary",
                num_classes=2,
                row_names=data_name,
                col_names=data_name,
            ).to(self.device)
            conf_matrices.append(confmat(label_preds[valid_mask], label_targets[valid_mask]))
        return conf_matrices


def _multi_class_cls_metric_callable(label_info: LabelInfo) -> MetricCollection:
    num_classes = label_info.num_classes
    if num_classes < 2:
        msg = (
            "Multiclass classification requires at least 2 classes. Use Multilabel classification "
            "for single class problems."
        )
        raise ValueError(msg)
    return MetricCollection(
        {
            "accuracy": TorchmetricAcc(task="multiclass", num_classes=num_classes),
            "f1-score": TorchmetricMulticlassF1(num_classes=num_classes, average="macro"),
        }
    )


MultiClassClsMetricCallable: MetricCallable = _multi_class_cls_metric_callable


def _multi_label_cls_metric_callable(label_info: LabelInfo) -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": MultilabelAccuracywithLabelGroup(label_info=label_info),
            "map": MultilabelmAP(label_info=label_info),
        },
    )


MultiLabelClsMetricCallable: MetricCallable = _multi_label_cls_metric_callable
