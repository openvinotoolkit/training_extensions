# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation of pose data sample."""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from otx.algo.utils.mmengine_utils import InstanceData

LABEL_TYPE = Union[torch.Tensor, np.ndarray, Sequence, int]


class PoseDataSample(InstanceData):
    """The base data structure of MMPose that is used as the interface between modules.

    The attributes of ``PoseDataSample`` includes:

        - ``keypoints``(np.ndarray): Keypoint annotations
        - ``keypoint_x_labels``(np.ndarray): Keypoint x-axis annotations according to simcc
        - ``keypoint_y_labels``(np.ndarray): Keypoint y-axis annotations according to simcc
        - ``keypoint_weights``(np.ndarray): Keypoint weight annotations from visibility
    """

    def __init__(self, *, metainfo: dict | None = None, **kwargs) -> None:
        super().__init__(metainfo=metainfo, **kwargs)
        self._keypoints: np.ndarray
        self._keypoint_x_labels: np.ndarray
        self._keypoint_y_labels: np.ndarray
        self._keypoint_weights: np.ndarray

    @property
    def keypoints(self) -> np.ndarray:
        """Property of `keypoints`."""
        return self._keypoints

    @keypoints.setter
    def keypoints(self, value: np.ndarray) -> None:
        """Setter of `keypoints`."""
        self.set_field(value, "_keypoints", dtype=np.ndarray)

    @keypoints.deleter
    def keypoints(self) -> None:
        """Deleter of `keypoints`."""
        del self._keypoints

    @property
    def keypoint_x_labels(self) -> np.ndarray:
        """Property of `keypoint_x_labels`."""
        return self._keypoint_x_labels

    @keypoint_x_labels.setter
    def keypoint_x_labels(self, value: np.ndarray) -> None:
        """Setter of `keypoint_x_labels`."""
        self.set_field(value, "_keypoint_x_labels", dtype=np.ndarray)

    @keypoint_x_labels.deleter
    def keypoint_x_labels(self) -> None:
        """Deleter of `keypoint_x_labels`."""
        del self._keypoint_x_labels

    @property
    def keypoint_y_labels(self) -> np.ndarray:
        """Property of `keypoint_y_labels`."""
        return self._keypoint_y_labels

    @keypoint_y_labels.setter
    def keypoint_y_labels(self, value: np.ndarray) -> None:
        """Setter of `keypoint_y_labels`."""
        self.set_field(value, "_keypoint_y_labels", dtype=np.ndarray)

    @keypoint_y_labels.deleter
    def keypoint_y_labels(self) -> None:
        """Deleter of `keypoint_y_labels`."""
        del self._keypoint_y_labels

    @property
    def keypoint_weights(self) -> np.ndarray:
        """Property of `keypoint_weights`."""
        return self._keypoints

    @keypoint_weights.setter
    def keypoint_weights(self, value: np.ndarray) -> None:
        """Setter of `keypoint_weights`."""
        self.set_field(value, "_keypoint_weights", dtype=np.ndarray)

    @keypoint_weights.deleter
    def keypoint_weights(self) -> None:
        """Deleter of `keypoint_weights`."""
        del self._keypoint_weights
