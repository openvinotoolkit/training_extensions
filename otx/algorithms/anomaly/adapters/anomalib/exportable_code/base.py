"""Wrapper for Open Model Zoo for Anomaly tasks."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Union

import numpy as np
from openvino.model_zoo.model_api.models import SegmentationModel
from openvino.model_zoo.model_api.models.types import NumericalValue


class AnomalyBase(SegmentationModel):
    """Wrapper for anomaly tasks."""

    __model__ = "anomaly_base"

    @classmethod
    def parameters(cls):
        """Dictionary containing model parameters."""
        parameters = super().parameters()
        parameters["resize_type"].update_default_value("standard")
        parameters.update(
            {
                "image_threshold": NumericalValue(description="Image-level Threshold value to locate anomaly"),
                "pixel_threshold": NumericalValue(description="Pixel-level Threshold value to locate anomaly"),
                "min": NumericalValue(description="Threshold value to locate anomaly"),
                "max": NumericalValue(description="Threshold value to locate anomaly"),
                "threshold": NumericalValue(description="Threshold used to classify anomaly"),
            }
        )

        return parameters

    @staticmethod
    def _normalize(
        targets: Union[np.ndarray, np.float32],
        threshold: Union[np.ndarray, float],
        min_val: Union[np.ndarray, float],
        max_val: Union[np.ndarray, float],
    ) -> np.ndarray:
        """Apply min-max normalization and shift the values such that the threshold value is centered at 0.5."""
        normalized = ((targets - threshold) / (max_val - min_val)) + 0.5
        if isinstance(targets, (np.ndarray, np.float32)):
            normalized = np.minimum(normalized, 1)
            normalized = np.maximum(normalized, 0)
        else:
            raise ValueError(f"Targets must be either Tensor or Numpy array. Received {type(targets)}")
        return normalized
