# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for classification validator metric reporting (F1, iter_time)."""

from __future__ import annotations

import pytest
import torch

from getitune.backend.ultralytics.validators.classification import (
    ClassificationValidator,
    MultiLabelClassificationValidator,
    _UltralyticsClassificationValidator,
)
from getitune.metrics.accuracy import MultiLabelClsMetricCallable
from getitune.types.label import LabelInfo


class TestClassificationValidatorGetStats:
    """Tests for the multi-class validator's ``get_stats`` override."""

    def test_adds_f1_score_and_iter_time_to_upstream_stats(self, mocker) -> None:
        """F1-score and iter_time must be merged alongside upstream top1/top5 accuracy."""
        validator = object.__new__(ClassificationValidator)
        validator.nc = 2
        # Top-2 sorted predictions per sample; column 0 is the top-1 prediction.
        validator.pred = [torch.tensor([[0, 1], [1, 0]])]
        validator.targets = [torch.tensor([0, 1])]
        validator._batch_times = [1.0, 0.2, 0.4]  # first entry is warmup, excluded from the mean

        mocker.patch.object(
            _UltralyticsClassificationValidator,
            "get_stats",
            return_value={"metrics/accuracy_top1": 1.0, "metrics/accuracy_top5": 1.0},
        )

        stats = validator.get_stats()

        assert stats["metrics/accuracy_top1"] == 1.0
        assert stats["metrics/f1-score"] == pytest.approx(1.0)
        assert stats["metrics/iter_time"] == pytest.approx(0.3)

    def test_iter_time_is_zero_without_recorded_batches(self, mocker) -> None:
        """No recorded batch times should yield 0.0 rather than raising."""
        validator = object.__new__(ClassificationValidator)
        validator.nc = 2
        validator.pred = []
        validator.targets = []

        mocker.patch.object(_UltralyticsClassificationValidator, "get_stats", return_value={})

        stats = validator.get_stats()

        assert stats["metrics/iter_time"] == 0.0
        assert "metrics/f1-score" not in stats


class TestMultiLabelClassificationValidatorGetStats:
    """Tests for the multi-label validator's ``get_stats`` override."""

    def test_includes_iter_time_alongside_accuracy_and_map(self) -> None:
        validator = object.__new__(MultiLabelClassificationValidator)
        label_info = LabelInfo(label_names=["a", "b"], label_ids=["0", "1"], label_groups=[["a", "b"]])
        validator.metric = MultiLabelClsMetricCallable(label_info)
        validator.metric.update(
            preds=torch.tensor([[0.9, 0.1], [0.2, 0.9]]),
            target=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )
        validator._batch_times = [1.0, 0.5]

        stats = validator.get_stats()

        assert stats["metrics/accuracy"] == pytest.approx(1.0)
        assert stats["metrics/map"] == pytest.approx(1.0)
        assert stats["metrics/iter_time"] == pytest.approx(0.5)
