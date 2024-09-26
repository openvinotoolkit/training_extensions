# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

if TYPE_CHECKING:
    from otx.core.types.label import LabelInfo


class MultilabelmAP(Metric):
    """Accuracy class for the multi-label classification with label_group.

    For the multi-label classification, the number of label_groups should be the same with number of labels.
    All lable_group represents whether the label exist or not (binary classification).
    """

    def __init__(
        self,
        label_info: LabelInfo,
    ):
        super().__init__()
        self.label_info: LabelInfo = label_info

        self.preds: list[Tensor] = []
        self.targets: list[Tensor] = []

    def reset(self) -> None:
        """Reset for every validation and test epoch.

        Please be careful that some variables should not be reset for each epoch.
        """
        super().reset()
        self.preds = []
        self.targets = []

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.preds.extend(preds)
        self.targets.extend(target)

    def compute(self) -> Tensor | dict[str, Any]:
        """Compute the metric."""
        metric_value = _map(torch.stack(self.targets).cpu().numpy(), torch.stack(self.preds).cpu().numpy())

        return {"mAP": Tensor([metric_value])}


def _map(targs: np.ndarray, preds: np.ndarray, pos_thr: float = 0.5) -> float:
    """Computes multi-label mAP metric."""

    def average_precision(output: np.ndarray, target: np.ndarray) -> float:
        epsilon = 1e-8

        # sort examples
        indices = output.argsort()[::-1]
        # Computes prec@i
        total_count_ = np.cumsum(np.ones((len(output), 1)))

        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)

        return precision_at_i_ / (total + epsilon)

    if np.size(preds) == 0:
        return 0
    ap = np.zeros(preds.shape[1])
    # compute average precision for each class
    for k in range(preds.shape[1]):
        scores = preds[:, k]
        targets = targs[:, k]
        ap[k] = average_precision(scores, targets)

    tp, fp, fn, tn = [], [], [], []
    for k in range(preds.shape[0]):
        scores = preds[k, :]
        targets = targs[k, :]
        pred = (scores > pos_thr).astype(np.int32)
        tp.append(((pred + targets) == 2).sum())
        fp.append(((pred - targets) == 1).sum())
        fn.append(((pred - targets) == -1).sum())
        tn.append(((pred + targets) == 0).sum())

    return ap.mean()
