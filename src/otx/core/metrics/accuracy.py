# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import torch
from torch import nn
from torchmetrics import ConfusionMatrix, Metric
from torchmetrics.classification.accuracy import Accuracy as TorchmetricAcc
from torchmetrics.classification.accuracy import (
    MultilabelAccuracy as TorchmetricMultilabelAcc,
)
from torchmetrics.collections import MetricCollection

from otx.core.metrics.types import MetricCallable

if TYPE_CHECKING:
    from torch import Tensor

    from otx.core.types.label import HLabelInfo, LabelInfo


class NamedConfusionMatrix(ConfusionMatrix):
    """Named Confusion Matrix to add row, col label names."""

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
        **kwargs: Any,  # noqa: ANN401
    ) -> NamedConfusionMatrix:
        """Construct the NamedConfusionMatrix."""
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

    @property
    def col_names(self) -> list[str]:
        """The names of colum."""
        return self.col_names

    @property
    def row_names(self) -> list[str]:
        """The names of row."""
        return self.row_names


class AccuracywithLabelGroup(Metric):
    """Base accuracy class for the OTX classification tasks with lable group.

    It calculates the accuracy with the label_groups information, not class.
    It means that average will be applied to the results from the each label groups.
    """

    def __init__(
        self,
        label_info: LabelInfo,
        *,
        average: Literal["MICRO", "MACRO"] = "MICRO",
        threshold: float = 0.5,
    ):
        super().__init__()
        self.average = average
        self.threshold = threshold
        self._label_info: LabelInfo = label_info

        self.preds: list[Tensor] = []
        self.targets: list[Tensor] = []

    @property
    def label_info(self) -> LabelInfo:
        """Get the member `AccuracywithLabelGroup` label information."""
        return self._label_info

    @label_info.setter
    def label_info(self, label_info: LabelInfo) -> None:
        self._label_info = label_info

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.extend(preds)
        self.targets.extend(target)

    def _compute_unnormalized_confusion_matrices(self) -> list[NamedConfusionMatrix]:
        raise NotImplementedError

    def _compute_accuracy_from_conf_matrices(self, conf_matrices: Tensor) -> Tensor:
        """Compute the accuracy from the confusion matrix."""
        correct_per_label_group = torch.stack([torch.trace(conf_matrix) for conf_matrix in conf_matrices])
        total_per_label_group = torch.stack([torch.sum(conf_matrix) for conf_matrix in conf_matrices])

        if self.average == "MICRO":
            return torch.sum(correct_per_label_group) / torch.sum(total_per_label_group)
        if self.average == "MACRO":
            return torch.nanmean(torch.divide(correct_per_label_group, total_per_label_group))

        msg = f"Average should be MICRO or MACRO, got {self.average}"
        raise ValueError(msg)

    def compute(self) -> Tensor | dict[str, Any]:
        """Compute the metric."""
        conf_matrices = self._compute_unnormalized_confusion_matrices()

        return {
            "conf_matrix": conf_matrices,
            "accuracy": self._compute_accuracy_from_conf_matrices(conf_matrices),
        }


class MulticlassAccuracywithLabelGroup(AccuracywithLabelGroup):
    """Accuracy class for the multi-class classification with label group.

    For the multi-class classification, the number of label_groups should be 1.
    So, the results always the same regardless of average method.
    """

    def _compute_unnormalized_confusion_matrices(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
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

            num_classes = len(label_group)
            confmat = NamedConfusionMatrix(
                task="multiclass",
                num_classes=num_classes,
                row_names=label_group,
                col_names=label_group,
            )
            conf_matrices.append(confmat(valid_preds, valid_targets))
        return conf_matrices


class MultilabelAccuracywithLabelGroup(AccuracywithLabelGroup):
    """Accuracy class for the multi-label classification with label_group.

    For the multi-label classification, the number of label_groups should be the same with number of labels.
    All lable_group represents whether the label exist or not (binary classification).
    """

    def _compute_unnormalized_confusion_matrices(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        preds = torch.stack(self.preds)
        targets = torch.stack(self.targets)

        conf_matrices = []
        for i, label_group in enumerate(self.label_info.label_groups):
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


class HlabelAccuracy(AccuracywithLabelGroup):
    """Accuracy class for the hierarchical-label classification.

    H-label Classification is the combination version of multi-class and multi-label classification.
    It could have multiple heads for the multi-class classification to classify complex hierarchy architecture.
    For the multi-label part, it's the same with the CusotmMultilabelAccuracy.
    """

    def _is_multiclass_group(self, label_group: list[str]) -> bool:
        return len(label_group) != 1

    def _compute_unnormalized_confusion_matrices(self) -> list[NamedConfusionMatrix]:
        """Compute an unnormalized confusion matrix for every label group."""
        preds = torch.stack(self.preds)
        targets = torch.stack(self.targets)

        conf_matrices = []
        for i, label_group in enumerate(self.label_info.label_groups):
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
    """Mixed accuracy metric for h-label classification.

    It only used multi-class and multi-label metrics from torchmetrics.
    This is different from the CustomHlabelAccuracy since MixedHLabelAccuracy doesn't use label_groups info.
    It makes large gap to the results since CusotmHlabelAccuracy averages the results by using the label_groups info.

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
        # https://github.com/Lightning-AI/torchmetrics/blob/6377aa5b6fe2863761839e6b8b5a857ef1b8acfa/src/torchmetrics/functional/classification/stat_scores.py#L583-L584
        # MultilabelAccuracy is available when num_multilabel_classes is greater than 2.
        self.multilabel_accuracy = None
        if self.num_multilabel_classes > 1:
            self.multilabel_accuracy = TorchmetricMultilabelAcc(
                num_labels=self.num_multilabel_classes,
                threshold=0.5,
                average="macro",
            )
        elif self.num_multilabel_classes == 1:
            self.multilabel_accuracy = TorchmetricAcc(task="binary", num_classes=self.num_multilabel_classes)

    def _apply(self, fn: Callable, exclude_state: Sequence[str] = "") -> nn.Module:
        self.multiclass_head_accuracy = [
            acc._apply(  # noqa: SLF001
                fn,
                exclude_state,
            )
            for acc in self.multiclass_head_accuracy
        ]
        if self.multilabel_accuracy is not None:
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

        if self.multilabel_accuracy is not None:
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

        if self.multilabel_accuracy is not None:
            multilabel_acc = self.multilabel_accuracy.compute()

            return (multiclass_accs + multilabel_acc) / 2

        return multiclass_accs


def _multi_class_cls_metric_callable(label_info: LabelInfo) -> MetricCollection:
    num_classes = label_info.num_classes
    task = "binary" if num_classes == 1 else "multiclass"
    return MetricCollection(
        {"accuracy": TorchmetricAcc(task=task, num_classes=num_classes)},
    )


MultiClassClsMetricCallable: MetricCallable = _multi_class_cls_metric_callable


def _multi_label_cls_metric_callable(label_info: LabelInfo) -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": MultilabelAccuracywithLabelGroup(label_info=label_info),
        },
    )


MultiLabelClsMetricCallable: MetricCallable = _multi_label_cls_metric_callable


def _mixed_hlabel_accuracy(label_info: HLabelInfo) -> MetricCollection:
    return MetricCollection(
        {
            "accuracy": MixedHLabelAccuracy(
                num_multiclass_heads=label_info.num_multiclass_heads,
                num_multilabel_classes=label_info.num_multilabel_classes,
                head_logits_info=label_info.head_idx_to_logits_range,
            ),
        },
    )


HLabelClsMetricCallable: MetricCallable = _mixed_hlabel_accuracy  # type: ignore[assignment]
