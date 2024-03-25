# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Test of Module for OTX custom metrices."""

from __future__ import annotations

import pytest
import torch
from otx.core.metrics.fmeasure import FMeasure
from otx.core.types.label import LabelInfo


class TestFMeasure:
    @pytest.fixture()
    def fxt_preds(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.Tensor([[0.7, 0.6, 0.9, 0.6], [0.2, 0.5, 0.8, 0.6]]),
                "labels": torch.IntTensor([0, 0]),
                "scores": torch.Tensor([0.9, 0.8]),
            },
            {
                "boxes": torch.Tensor([[0.3, 0.4, 0.6, 0.6], [0.3, 0.3, 0.4, 0.5]]),
                "labels": torch.IntTensor([0, 0]),
                "scores": torch.Tensor([0.9, 0.8]),
            },
        ]

    @pytest.fixture()
    def fxt_targets(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.Tensor([[0.8, 0.6, 0.9, 0.7], [0.3, 0.5, 0.8, 0.7]]),
                "labels": torch.IntTensor([0, 0]),
            },
            {
                "boxes": torch.Tensor([[0.4, 0.4, 0.6, 0.6], [0.3, 0.3, 0.4, 0.4]]),
                "labels": torch.IntTensor([0, 0]),
            },
        ]

    def test_fmeasure(self, fxt_preds, fxt_targets) -> None:
        """Check whether f1 score is same with OTX1.x version."""
        metric = FMeasure(label_info=LabelInfo.from_num_classes(1))
        metric.update(fxt_preds, fxt_targets)
        result = metric.compute()
        assert result["f1-score"] == 0.5
        best_confidence_threshold = metric.best_confidence_threshold
        assert isinstance(best_confidence_threshold, float)

        metric.reset()
        assert metric.preds == []
        assert metric.targets == []

        # TODO(jaegukhyun): Add the following scenario
        # 1. Prepare preds and targets which can produce f1-score < 0.5
        # 2. Execute metric.compute()
        # 3. Assert best_confidence_threshold == metric.best_confidence_threshold

    def test_fmeasure_with_fixed_threshold(self, fxt_preds, fxt_targets) -> None:
        """Check fmeasure can compute f1 score given confidence threshold."""
        metric = FMeasure(label_info=LabelInfo.from_num_classes(1))

        metric.update(fxt_preds, fxt_targets)
        result = metric.compute(best_confidence_threshold=0.85)
        assert result["f1-score"] == 0.3333333432674408
