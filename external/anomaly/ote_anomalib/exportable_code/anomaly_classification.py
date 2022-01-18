"""Wrapper for Open Model Zoo for Anomaly tasks."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict, Union

import cv2
import numpy as np
from openvino.model_zoo.model_api.models import SegmentationModel
from openvino.model_zoo.model_api.models.types import NumericalValue


class AnomalyClassification(SegmentationModel):
    """Wrapper for anomaly classification task."""

    __model__ = "anomaly_classification"

    @classmethod
    def parameters(cls):
        """Dictionary containing model parameters."""
        parameters = super().parameters()
        parameters["resize_type"].update_default_value("standard")
        parameters.update(
            {
                "image_threshold": NumericalValue(description="Threshold value to locate anomaly"),
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

    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]) -> float:
        """Resize the outputs of the model to original image size.

        Args:
            outputs (Dict[str, np.ndarray]): Raw outputs of the model after ``infer_sync`` is called.
            meta (Dict[str, Any]): Metadata which contains values such as threshold, original image size.

        Returns:
            float: Normalized anomaly score
        """
        anomaly_map: np.ndarray = outputs[self.output_blob_name].squeeze()
        pred_score = anomaly_map.reshape(-1).max()

        meta["image_threshold"] = self.image_threshold  # pylint: disable=no-member
        meta["min"] = self.min  # pylint: disable=no-member
        meta["max"] = self.max  # pylint: disable=no-member
        meta["threshold"] = self.threshold  # pylint: disable=no-member

        anomaly_map = self._normalize(anomaly_map, meta["image_threshold"], meta["min"], meta["max"])
        pred_score = self._normalize(pred_score, meta["image_threshold"], meta["min"], meta["max"])

        input_image_height = meta["original_shape"][0]
        input_image_width = meta["original_shape"][1]
        result = cv2.resize(anomaly_map, (input_image_width, input_image_height))

        meta["anomaly_map"] = result

        return np.array(pred_score)
