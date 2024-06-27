# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.losses.accuracy.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/losses/accuracy.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def accuracy(
    pred: Tensor,
    target: Tensor,
    topk: int | tuple[int] = 1,
    thresh: float | None = None,
) -> list[Tensor] | Tensor:
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (Tensor): The model prediction, shape (N, num_class)
        target (Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        (float | tuple[float]): If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    if not isinstance(topk, (int, tuple)):
        msg = f"topk must be int or tuple of int, got {type(topk)}"
        raise TypeError(msg)
    if isinstance(topk, int):
        topk = (topk,)
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.0) for i in range(len(topk))]
        return accu[0] if return_single else accu
    if pred.ndim != 2 or target.ndim != 1:
        msg = "Input tensors must have 2 dims for pred and 1 dim for target"
        raise ValueError(msg)
    if pred.size(0) != target.size(0):
        msg = "Input tensors must have the same size along the 0th dim"
        raise ValueError(msg)
    if maxk > pred.size(1):
        msg = f"maxk {maxk} exceeds pred dimension {pred.size(1)}"
        raise ValueError(msg)
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res
