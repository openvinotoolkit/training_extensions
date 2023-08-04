"""Wrapper for Open Model Zoo for Anomaly Classification tasks."""

# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Dict

import cv2
import numpy as np

from .base import AnomalyBase


class AnomalyClassification(AnomalyBase):
    """Wrapper for anomaly classification task."""

    __model__ = "anomaly_classification"

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

        meta["image_threshold"] = self.metadata["image_threshold"]  # pylint: disable=no-member
        meta["min"] = self.metadata["min"]  # pylint: disable=no-member
        meta["max"] = self.metadata["max"]  # pylint: disable=no-member
        meta["threshold"] = self.threshold  # pylint: disable=no-member

        anomaly_map = self._normalize(anomaly_map, meta["image_threshold"], meta["min"], meta["max"])
        pred_score = self._normalize(pred_score, meta["image_threshold"], meta["min"], meta["max"])

        input_image_height = meta["original_shape"][0]
        input_image_width = meta["original_shape"][1]
        result = cv2.resize(anomaly_map, (input_image_width, input_image_height))

        meta["anomaly_map"] = result

        return np.array(pred_score)
