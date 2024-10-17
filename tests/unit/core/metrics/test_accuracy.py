# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of Module for OTX custom metrices."""

import pytest
import torch
from otx.core.metrics.accuracy import (
    HlabelAccuracy,
    MixedHLabelAccuracy,
    MulticlassAccuracywithLabelGroup,
    MultiClassClsMetricCallable,
    MultilabelAccuracywithLabelGroup,
)
from otx.core.types.label import HLabelInfo, LabelInfo
from torchmetrics.classification.accuracy import BinaryAccuracy, MulticlassAccuracy, MultilabelAccuracy


class TestAccuracy:
    def test_multiclass_accuracy(self, fxt_multiclass_labelinfo: LabelInfo) -> None:
        """Check whether accuracy is same with OTX1.x version."""
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

    def test_default_multi_class_cls_metric_callable(self, fxt_multiclass_labelinfo: LabelInfo) -> None:
        assert fxt_multiclass_labelinfo.num_classes > 1
        metric = MultiClassClsMetricCallable(fxt_multiclass_labelinfo)
        assert isinstance(metric.accuracy, MulticlassAccuracy)

        one_class_label_info = LabelInfo(label_names=["class1"], label_groups=[["class1"]])
        assert one_class_label_info.num_classes == 1
        binary_metric = MultiClassClsMetricCallable(one_class_label_info)
        assert isinstance(binary_metric.accuracy, BinaryAccuracy)

    def test_multilabel_accuracy(self, fxt_multilabel_labelinfo: LabelInfo) -> None:
        """Check whether accuracy is same with OTX1.x version."""
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

    def test_hlabel_accuracy(self, fxt_hlabel_multilabel_info: HLabelInfo) -> None:
        """Check whether accuracy is same with OTX1.x version."""
        preds = [
            torch.Tensor([1, -1, 0, 0.2, 0.8, 0.9]),
            torch.Tensor([1, 0, 0, 0.8, 0.7, 0.7]),
        ]
        targets = [
            torch.Tensor([1, -1, 0, 0, 1, 1]),
            torch.Tensor([0, 0, 1, 0, 1, 0]),
        ]

        metric = HlabelAccuracy(fxt_hlabel_multilabel_info, average="MICRO")
        metric.update(preds, targets)
        result = metric.compute()
        acc = result["accuracy"]
        assert round(acc.item(), 3) == 0.636


class TestMixedHLabelAccuracy:
    @pytest.fixture()
    def hlabel_accuracy(self) -> MixedHLabelAccuracy:
        # You may need to adjust the parameters based on your actual use case
        return MixedHLabelAccuracy(
            num_multiclass_heads=2,
            num_multilabel_classes=3,
            head_logits_info={"head1": (0, 5), "head2": (5, 10)},
            threshold_multilabel=0.5,
        )

    def test_update_and_compute(self, hlabel_accuracy) -> None:
        preds = torch.rand((10, 5))
        target = torch.randint(0, 2, (10, 5))  # Replace the dimensions with actual dimensions

        hlabel_accuracy.update(preds, target)
        result = hlabel_accuracy.compute()

        assert isinstance(result, torch.Tensor)

    def test_multilabel_only(self) -> None:
        # Test when only multilabel heads are present (should raise an exception)
        with pytest.raises(ValueError, match="The number of multiclass heads should be larger than 0"):
            MixedHLabelAccuracy(
                num_multiclass_heads=0,
                num_multilabel_classes=3,
                head_logits_info={"head1": (0, 5), "head2": (5, 10)},
                threshold_multilabel=0.5,
            )

    def test_multilabel_accuracy(self, hlabel_accuracy) -> None:
        # Normal Case: num_multilabel_classes > 1 -> MultilabelAccuracy
        assert hlabel_accuracy.num_multilabel_classes == 3
        assert isinstance(hlabel_accuracy.multilabel_accuracy, MultilabelAccuracy)

        # Edge Case: num_multilabel_classes = 1 -> BinaryAccuracy
        acc = MixedHLabelAccuracy(
            num_multiclass_heads=2,
            num_multilabel_classes=1,
            head_logits_info={"head1": (0, 5), "head2": (5, 10)},
            threshold_multilabel=0.5,
        )
        assert acc.num_multilabel_classes == 1
        assert isinstance(acc.multilabel_accuracy, BinaryAccuracy)

        # None Case: num_multilabel_classes = 0 -> None
        acc = MixedHLabelAccuracy(
            num_multiclass_heads=2,
            num_multilabel_classes=0,
            head_logits_info={"head1": (0, 5), "head2": (5, 10)},
            threshold_multilabel=0.5,
        )
        assert acc.num_multilabel_classes == 0
        assert acc.multilabel_accuracy is None
