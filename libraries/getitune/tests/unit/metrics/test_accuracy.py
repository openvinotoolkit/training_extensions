# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of Module for getitune custom metrices."""

import pytest
import torch
from torchmetrics.classification.accuracy import MulticlassAccuracy

from getitune.metrics.accuracy import (
    MulticlassAccuracywithLabelGroup,
    MultiClassClsMetricCallable,
    MultilabelAccuracywithLabelGroup,
)
from getitune.types.label import LabelInfo


class TestAccuracy:
    def test_multiclass_accuracy(self, fxt_multiclass_labelinfo: LabelInfo) -> None:
        """Check whether accuracy is same with getitune 1.x version."""
        preds = [
            torch.Tensor([0]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            torch.Tensor([1]),
            torch.Tensor([2]),
            torch.Tensor([2]),
        ]
        targets = [
            torch.Tensor([0]),
            torch.Tensor([0]),
            torch.Tensor([1]),
            torch.Tensor([1]),
            torch.Tensor([1]),
            torch.Tensor([2]),
        ]
        metric = MulticlassAccuracywithLabelGroup(fxt_multiclass_labelinfo, average="MICRO")
        metric.update(preds, targets)
        result = metric.compute()
        acc = result["accuracy"]
        assert round(acc.item(), 3) == 0.800

        metric = MulticlassAccuracywithLabelGroup(fxt_multiclass_labelinfo, average="MACRO")
        metric.update(preds, targets)
        result = metric.compute()
        acc = result["accuracy"]
        assert round(acc.item(), 3) == 0.792

        metric_collection = MultiClassClsMetricCallable(fxt_multiclass_labelinfo)
        assert isinstance(metric_collection.accuracy, MulticlassAccuracy)

        preds_tensor = torch.tensor([0, 1, 2, 2])
        targets_tensor = torch.tensor([0, 1, 1, 2])
        metric_collection.update(preds_tensor, targets_tensor)
        result = metric_collection.compute()
        assert result["f1-score"].item() == pytest.approx(0.7777778, rel=1e-5)

    def test_single_class_multiclass_metric_rejected(self) -> None:
        label_info = LabelInfo(label_names=["class1"], label_groups=[["class1"]], label_ids=["0"])
        with pytest.raises(ValueError, match="Multiclass classification requires at least 2 classes"):
            MultiClassClsMetricCallable(label_info)

    def test_multilabel_accuracy(self, fxt_multilabel_labelinfo: LabelInfo) -> None:
        """Check whether accuracy is same with getitune 1.x version."""
        preds = [
            torch.Tensor([0.2, 0.8, 0.9]),
            torch.Tensor([0.8, 0.7, 0.7]),
        ]
        targets = [
            torch.Tensor([0, 1, 1]),
            torch.Tensor([0, 1, 0]),
        ]
        metric = MultilabelAccuracywithLabelGroup(fxt_multilabel_labelinfo, average="MICRO")
        metric.update(preds, targets)
        result = metric.compute()
        acc = result["accuracy"]
        assert round(acc.item(), 3) == 0.667

    def test_multilabel_accuracy_with_single_combined_label_group(self) -> None:
        """Multi-label ``label_info`` normally holds all labels in one combined group

        (see ``MultilabelClsDataset.label_info`` / ``UltralyticsModel._dispatch_label_info``).
        The metric must still evaluate every individual label column, not just
        as many columns as there are entries in ``label_groups``.
        """
        label_names = ["class1", "class2", "class3"]
        label_info = LabelInfo(
            label_names=label_names,
            label_groups=[label_names],
            label_ids=["0", "1", "2"],
        )
        preds = [
            torch.Tensor([0.2, 0.8, 0.9]),
            torch.Tensor([0.8, 0.7, 0.7]),
        ]
        targets = [
            torch.Tensor([0, 1, 1]),
            torch.Tensor([0, 1, 0]),
        ]
        metric = MultilabelAccuracywithLabelGroup(label_info, average="MICRO")
        metric.update(preds, targets)
        result = metric.compute()
        assert isinstance(result, dict)
        acc = result["accuracy"]
        assert round(acc.item(), 3) == 0.667
        assert len(result["conf_matrix"]) == len(label_names)
