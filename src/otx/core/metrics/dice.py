# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX Dice metric used for the OTX semantic segmentation task."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torchmetrics import JaccardIndex
from torchmetrics.classification.dice import Dice
from torchmetrics.collections import MetricCollection

from otx.core.types.label import SegLabelInfo

if TYPE_CHECKING:
    from torch import Tensor


def _segm_callable(label_info: SegLabelInfo) -> MetricCollection:
    return MetricCollection(
        {
            "Dice": OTXDice(num_classes=label_info.num_classes, ignore_index=label_info.ignore_index, average="macro"),
            "mIoU": JaccardIndex(
                task="multiclass",
                num_classes=label_info.num_classes,
                ignore_index=label_info.ignore_index,
            ),
        },
    )


class OTXDice(Dice):
    """Dice metric used for the OTX semantic segmentation task."""

    def __init__(
        self,
        zero_division: int = 0,
        num_classes: int | None = None,
        threshold: float = 0.5,
        average: Literal["micro", "macro", "none"] = "micro",
        mdmc_average: str = "global",
        ignore_index: int | None = None,
        top_k: int | None = None,
        multiclass: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            zero_division=zero_division,
            num_classes=num_classes,
            threshold=threshold,
            average=average,
            mdmc_average=mdmc_average,
            ignore_index=None,
            top_k=top_k,
            multiclass=multiclass,
            **kwargs,
        )
        # workaround to use ignore index > num_classes or < 0
        self.extended_ignore_index = ignore_index

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets. Fix ignore_index handling."""
        if self.extended_ignore_index is not None:
            filtered_preds = preds[target != self.extended_ignore_index]
            filtered_target = target[target != self.extended_ignore_index]
        else:
            filtered_preds = preds
            filtered_target = target

        super().update(filtered_preds, filtered_target)


SegmCallable = _segm_callable
