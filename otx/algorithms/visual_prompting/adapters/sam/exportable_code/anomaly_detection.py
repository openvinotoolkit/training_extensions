"""Wrapper for Open Model Zoo for Anomaly Detection tasks."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict

import cv2
import numpy as np

from .base import AnomalyBase


class AnomalyDetection(AnomalyBase):
    """Wrapper for anomaly detection task."""

    __model__ = "anomaly_detection"

    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]) -> np.ndarray:
        """Resize the outputs of the model to original image size.

        Args:
            outputs (Dict[str, np.ndarray]): Raw outputs of the model after ``infer_sync`` is called.
            meta (Dict[str, Any]): Metadata which contains values such as threshold, original image size.

        Returns:
            np.ndarray: Detection Mask
        """
        anomaly_map: np.ndarray = outputs[self.output_blob_name].squeeze()

        meta["pixel_threshold"] = self.pixel_threshold  # pylint: disable=no-member
        meta["min"] = self.min  # pylint: disable=no-member
        meta["max"] = self.max  # pylint: disable=no-member
        meta["threshold"] = self.threshold  # pylint: disable=no-member

        anomaly_map = self._normalize(anomaly_map, meta["pixel_threshold"], meta["min"], meta["max"])

        input_image_height = meta["original_shape"][0]
        input_image_width = meta["original_shape"][1]
        result = cv2.resize(anomaly_map, (input_image_width, input_image_height))

        return result
