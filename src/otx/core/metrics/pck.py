# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX accuracy metric used for classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor
from torchmetrics import Metric

if TYPE_CHECKING:
    from otx.core.types.label import LabelInfo


def _calc_distances(preds: np.ndarray, gts: np.ndarray, mask: np.ndarray, norm_factor: np.ndarray) -> np.ndarray:
    """Calculate the normalized distances between preds and target.

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        gts (np.ndarray[N, K, D]): Ground truth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        norm_factor (np.ndarray[N, D]): Normalization factor.
            Typical value is heatmap_size.

    Returns:
        np.ndarray[K, N]: The normalized distances. \
            If target keypoints are missing, the distance is -1.
    """
    batch_size, num_keypoints, _ = preds.shape
    # set mask=0 when norm_factor==0
    _mask = mask.copy()
    _mask[np.where((norm_factor == 0).sum(1))[0], :] = False

    distances = np.full((batch_size, num_keypoints), -1, dtype=np.float32)
    # handle invalid values
    norm_factor[np.where(norm_factor <= 0)] = 1e6
    distances[_mask] = np.linalg.norm(((preds - gts) / norm_factor[:, None, :])[_mask], axis=-1)
    return distances.T


def _distance_acc(distances: np.ndarray, thr: float = 0.5) -> float | int:
    """Return the percentage below the distance threshold.

    Ignore distances values with -1.

    Note:
        - instance number: N

    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold. \
            If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


def keypoint_pck_accuracy(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    thr: float,
    norm_factor: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Calculate the pose accuracy of PCK for each individual keypoint.

    And the averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        - instance number: N
        - keypoint number: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        norm_factor (np.ndarray[N, 2]): Normalization factor for the keypoints.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, norm_factor)
    acc: np.ndarray = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc: np.ndarray = acc[acc >= 0]
    cnt: int = len(valid_acc)
    avg_acc: float = valid_acc.mean() if cnt > 0 else 0.0
    return acc, avg_acc, cnt


class PCKMeasure(Metric):
    """Computes the pose accuracy (also known as PCK) for a resultset.

    Args:
        label_info (int): Dataclass including label information.
        dist_threshold (float): Threshold of PCK calculation.
    """

    def __init__(
        self,
        label_info: LabelInfo,
        dist_threshold: float = 0.05,
    ):
        super().__init__()

        self.label_info: LabelInfo = label_info
        self.dist_threshold: float = dist_threshold
        self.reset()

    @property
    def input_size(self) -> tuple[int, int]:
        """Getter for input_size."""
        return self._input_size

    @input_size.setter
    def input_size(self, size: tuple[int, int]) -> None:
        """Setter for input_size."""
        if not isinstance(size, tuple) or len(size) != 2:
            msg = "input_size must be a tuple of two integers."
            raise ValueError(msg)
        if not all(isinstance(dim, int) for dim in size):
            msg = "input_size dimensions must be integers."
            raise ValueError(msg)
        self._input_size = size

    def reset(self) -> None:
        """Reset for every validation and test epoch.

        Please be careful that some variables should not be reset for each epoch.
        """
        super().reset()
        self.preds: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

    def update(self, preds: list[dict[str, Tensor]], target: list[dict[str, Tensor]]) -> None:
        """Update total predictions and targets from given batch predicitons and targets."""
        for pred, tget in zip(preds, target):
            self.preds.extend(
                [
                    (pred["keypoints"], pred["scores"]),
                ],
            )
            self.targets.extend(
                [
                    (tget["keypoints"], tget["keypoints_visible"]),
                ],
            )

    def compute(self) -> dict:
        """Compute PCK score metric."""
        pred_kpts = np.stack([p[0].cpu().numpy() for p in self.preds])
        gt_kpts_processed = []
        for p in self.targets:
            if len(p[0].shape) == 3 and p[0].shape[0] == 1:
                gt_kpts_processed.append(p[0].squeeze())
            else:
                gt_kpts_processed.append(p[0])
        gt_kpts = np.stack(gt_kpts_processed)

        kpts_visible = []
        for p in self.targets:
            if len(p[1].shape) == 3 and p[1].shape[0] == 1:
                kpts_visible.append(p[1].squeeze())
            else:
                kpts_visible.append(p[1])

        kpts_visible_stacked = np.stack(kpts_visible)

        normalize = np.tile(np.array([self.input_size[::-1]]), (pred_kpts.shape[0], 1))
        _, avg_acc, _ = keypoint_pck_accuracy(
            pred_kpts,
            gt_kpts,
            mask=kpts_visible_stacked > 0,
            thr=self.dist_threshold,
            norm_factor=normalize,
        )

        return {"PCK": Tensor([avg_acc])}


def _pck_measure_callable(label_info: LabelInfo) -> PCKMeasure:
    return PCKMeasure(label_info=label_info)


PCKMeasureCallable = _pck_measure_callable
