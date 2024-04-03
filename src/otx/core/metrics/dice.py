# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX Dice metric used for the OTX semantic segmentation task."""
from torch import Tensor
from torchmetrics import JaccardIndex
from torchmetrics.classification.dice import Dice
from torchmetrics.collections import MetricCollection

from otx.core.types.label import LabelInfo


def _segm_callable(label_info: LabelInfo) -> MetricCollection:
    return MetricCollection(
        {
            "Dice": PatchedDice(num_classes=label_info.num_classes, ignore_index=-1, average="macro"),
            "mIoU": JaccardIndex(task="multiclass", num_classes=label_info.num_classes, ignore_index=255),
        },
    )


class PatchedDice(Dice):
    """Dice metric used for the OTX semantic segmentation task."""

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets. Fix ignore_index handling."""
        if self.ignore_index == -1 and self.num_classes < 255:
            # drop ignore index == 255
            filtered_preds = preds[target != 255]
            filtered_target = target[target != 255]
        super().update(filtered_preds, filtered_target)


SegmCallable = _segm_callable
