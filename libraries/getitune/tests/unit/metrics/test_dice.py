# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for getitune Dice metric."""

from __future__ import annotations

import torch
from torchmetrics.classification.jaccard import MulticlassJaccardIndex
from torchmetrics.collections import MetricCollection
from torchmetrics.segmentation.dice import DiceScore

from getitune.metrics.dice import DiceMetric, SegmCallable
from getitune.types.label import SegLabelInfo


class TestDice:
    def test_segm_callable_builds_expected_metrics(self, fxt_seg_label_info: SegLabelInfo) -> None:
        metric = SegmCallable(fxt_seg_label_info)

        assert isinstance(metric, MetricCollection)
        assert set(metric.keys()) == {"Dice", "mIoU"}
        assert isinstance(metric["Dice"], DiceMetric)
        assert isinstance(metric["mIoU"], MulticlassJaccardIndex)

    def test_perfect_prediction_returns_one(self) -> None:
        metric = DiceMetric(num_classes=3, average="macro", ignore_index=255)

        preds = torch.tensor([[[0, 1], [2, 1]]])
        target = torch.tensor([[[0, 1], [2, 1]]])

        metric.update(preds, target)
        score = metric.compute()

        assert torch.isclose(score, torch.tensor(1.0))

    def test_matches_torchmetrics_when_no_ignore_index(self) -> None:
        preds = torch.tensor([[[0, 1], [2, 1]]])
        target = torch.tensor([[[0, 1], [1, 2]]])

        dice_metric = DiceMetric(num_classes=3, average="macro", ignore_index=None)
        ref_metric = DiceScore(
            num_classes=3,
            average="macro",
            aggregation_level="global",
            input_format="index",
            include_background=False,
        )

        dice_metric.update(preds.clone(), target.clone())
        ref_metric.update(preds.clone().long(), target.clone().long())

        assert torch.allclose(dice_metric.compute(), ref_metric.compute())

    def test_ignore_index_excludes_ignored_pixels(self) -> None:
        metric = DiceMetric(num_classes=3, average="macro", ignore_index=255)

        # Bottom row is ignore_index; only top row should affect score.
        preds = torch.tensor([[[1, 2], [2, 1]]])
        target = torch.tensor([[[1, 2], [255, 255]]])

        metric.update(preds, target)
        score = metric.compute()

        # Remaining non-ignored pixels are predicted perfectly.
        assert torch.isclose(score, torch.tensor(1.0))

    def test_update_casts_float_inputs_to_long(self) -> None:
        metric = DiceMetric(num_classes=3, average="macro", ignore_index=255)

        preds = torch.tensor([[[0.0, 1.0], [2.0, 1.0]]], dtype=torch.float32)
        target = torch.tensor([[[0.0, 1.0], [2.0, 1.0]]], dtype=torch.float32)

        metric.update(preds, target)
        score = metric.compute()

        assert torch.isclose(score, torch.tensor(1.0))
