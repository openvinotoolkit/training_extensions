# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for getitune Dice metric used for the getitune semantic segmentation task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torchmetrics import JaccardIndex
from torchmetrics.collections import MetricCollection
from torchmetrics.segmentation.dice import DiceScore as _DiceScore

from getitune.types.label import SegLabelInfo

if TYPE_CHECKING:
    from torch import Tensor


def _segm_callable(label_info: SegLabelInfo) -> MetricCollection:
    return MetricCollection(
        {
            "Dice": DiceMetric(
                num_classes=label_info.num_classes, ignore_index=label_info.ignore_index, average="macro"
            ),
            "mIoU": JaccardIndex(
                task="multiclass",
                num_classes=label_info.num_classes,
                ignore_index=label_info.ignore_index,
            ),
        },
    )


class DiceMetric(_DiceScore):
    """Dice metric used for the getitune semantic segmentation task."""

    def __init__(
        self,
        num_classes: int | None = None,
        average: Literal["micro", "macro", "none"] = "macro",
        aggregation_level: Literal["global", "samplewise"] = "global",
        ignore_index: int | None = None,
        input_format: Literal["one-hot", "index", "mixed"] = "index",
        **kwargs,
    ) -> None:
        super().__init__(
            num_classes=num_classes,  # pyrefly: ignore[bad-argument-type]
            average=average,
            aggregation_level=aggregation_level,
            input_format=input_format,
            include_background=False,
            **kwargs,
        )
        self.ignore_index = ignore_index

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets. Fix ignore_index handling."""
        # treat ignore_index as a background to exclude it from metric computation
        if self.ignore_index is not None:
            mask = target == self.ignore_index
            if mask.any():
                # Work on copies to avoid mutating caller-provided tensors in-place
                preds = preds.clone()
                target = target.clone()
                preds[mask] = 0
                target[mask] = 0
        super().update(preds.long(), target.long())


SegmCallable = _segm_callable
