# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, Sequence

import torch
from torch import nn
from torchmetrics import ConfusionMatrix, Metric
from torchmetrics.classification.accuracy import Accuracy as TorchmetricAcc
from torchmetrics.classification.accuracy import MultilabelAccuracy as TorchmetricMultilabelAcc

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

    def _compute_unnormalized_confusion_matrices(self) -> list[NamedConfusionMatrix]:
        raise NotImplementedError

    def _compute_accuracy_from_conf_matrices(self, conf_matrices: list[NamedConfusionMatrix]) -> Tensor:
        """Compute the accuracy from the confusion matrix."""
        correct_per_label_group = torch.stack([torch.trace(conf_matrix) for conf_matrix in conf_matrices])
        total_per_label_group = torch.stack([torch.sum(conf_matrix) for conf_matrix in conf_matrices])

        if self.average == "MICRO":
            return torch.sum(correct_per_label_group) / torch.sum(total_per_label_group)
        if self.average == "MACRO":
            return torch.nanmean(torch.divide(correct_per_label_group, total_per_label_group))

        msg = f"Average should be MICRO or MACRO, got {self.average}"
        raise ValueError(msg)

    def compute(self) -> Tensor:
        """Compute the metric."""
        conf_matrices = self._compute_unnormalized_confusion_matrices()

        return {
            "conf_matrix": conf_matrices,
            "accuracy": self._compute_accuracy_from_conf_matrices(conf_matrices),
        }


class CustomMulticlassAccuracy(CustomAccuracy):
    """Custom accuracy class for the multi-class classification."""

    def _compute_unnormalized_confusion_matrices(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        conf_matrices = []
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
            conf_matrices.append(confmat(filtered_preds, filtered_targets))
        return conf_matrices


class CustomMultilabelAccuracy(CustomAccuracy):
    """Custom accuracy class for the multi-label classification."""

    def _compute_unnormalized_confusion_matrices(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        preds = torch.stack(self.preds)
        targets = torch.stack(self.targets)

        conf_matrices = []
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
            conf_matrices.append(confmat(valid_preds, valid_targets))
        return conf_matrices


class CustomHlabelAccuracy(CustomAccuracy):
    """Custom accuracy class for the hierarchical-label classification."""

    def _is_multiclass_group(self, label_group: list[str]) -> bool:
        return len(label_group) != 1

    def _compute_unnormalized_confusion_matrices(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        preds = torch.stack(self.preds)
        targets = torch.stack(self.targets)

        conf_matrices = []
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
                conf_matrices.append(confmat(valid_preds, valid_targets))
            else:
                label_preds = (label_preds >= self.threshold).long()
                data_name = [label_group[0], "~" + label_group[0]]
                confmat = NamedConfusionMatrix(
                    task="binary",
                    num_classes=2,
                    row_names=data_name,
                    col_names=data_name,
                ).to(self.device)
                conf_matrices.append(confmat(valid_preds, valid_targets))
        return conf_matrices


class MixedHLabelAccuracy(Metric):
    """Custom accuracy metric for h-label classification.

    Args:
        num_multiclass_heads (int): Number of multi-class heads.
        num_multilabel_classes (int): Number of multi-label classes.
        head_idx_to_logits_range (dict[str, tuple[int, int]]): The range of logits which represents
                                                                the number of classes for each heads.
        threshold_multilabel (float): Predictions with scores under the thresholds
                                        are considered as negative. Defaults to 0.5.
    """

    def __init__(
        self,
        num_multiclass_heads: int,
        num_multilabel_classes: int,
        head_logits_info: dict[str, tuple[int, int]],
        threshold_multilabel: float = 0.5,
    ):
        super().__init__()

        self.num_multiclass_heads = num_multiclass_heads
        if num_multiclass_heads == 0:
            msg = "The number of multiclass heads should be larger than 0"
            raise ValueError(msg)

        self.num_multilabel_classes = num_multilabel_classes
        self.threshold_multilabel = threshold_multilabel

        # Multiclass classification accuracy
        self.multiclass_head_accuracy: list[TorchmetricAcc] = [
            TorchmetricAcc(
                task="multiclass",
                num_classes=int(head_range[1] - head_range[0]),
            )
            for head_range in head_logits_info.values()
        ]

        # Multilabel classification accuracy metrics
        if self.num_multilabel_classes > 0:
            self.multilabel_accuracy = TorchmetricMultilabelAcc(
                num_labels=self.num_multilabel_classes,
                threshold=0.5,
                average="macro",
            )

    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> nn.Module:
        self.multiclass_head_accuracy = [acc._apply(fn, exclude_state) for acc in self.multiclass_head_accuracy]  # noqa: SLF001
        if self.num_multilabel_classes > 0:
            self.multilabel_accuracy = self.multilabel_accuracy._apply(fn, exclude_state)  # noqa: SLF001
        return self

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        # Split preds into multiclass and multilabel parts
        for head_idx in range(self.num_multiclass_heads):
            preds_multiclass = preds[:, head_idx]
            target_multiclass = target[:, head_idx]
            multiclass_mask = target_multiclass > 0

            is_all_multiclass_ignored = not multiclass_mask.any()
            if not is_all_multiclass_ignored:
                self.multiclass_head_accuracy[head_idx].update(
                    preds_multiclass[multiclass_mask],
                    target_multiclass[multiclass_mask],
                )

        if self.num_multilabel_classes > 0:
            # Split preds into multiclass and multilabel parts
            preds_multilabel = preds[:, self.num_multiclass_heads :]
            target_multilabel = target[:, self.num_multiclass_heads :]
            # Multilabel update
            self.multilabel_accuracy.update(preds_multilabel, target_multilabel)

    def compute(self) -> torch.Tensor:
        """Compute the final statistics."""
        multiclass_accs = torch.mean(
            torch.stack(
                [acc.compute() for acc in self.multiclass_head_accuracy],
            ),
        )

        if self.num_multilabel_classes > 0:
            multilabel_acc = self.multilabel_accuracy.compute()

            return (multiclass_accs + multilabel_acc) / 2

        return multiclass_accs
