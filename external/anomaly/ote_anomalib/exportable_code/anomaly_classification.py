"""Wrapper for Open Model Zoo for Anomaly tasks."""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

from typing import Any, Dict

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
                "pixel_threshold": NumericalValue(description="Threshold value to locate anomaly"),
                "min": NumericalValue(description="Threshold value to locate anomaly"),
                "max": NumericalValue(description="Threshold value to locate anomaly"),
            }
        )

        return parameters

    def postprocess(self, outputs: Dict[str, np.ndarray], meta: Dict[str, Any]) -> np.ndarray:
        """Resize the outputs of the model to original image size.

        Args:
            outputs (Dict[str, np.ndarray]): Raw outputs of the model after ``infer_sync`` is called.
            meta (Dict[str, Any]): Metadata which contains values such as threshold, original image size.

        Returns:
            np.ndarray: Resulting image resized to original input image size
        """
        outputs = outputs[self.output_blob_name].squeeze()
        anomaly_map = outputs[self.output_blob_name].squeeze()
        pred_score = anomaly_map.reshape(-1).max()

        meta["image_threshold"] = self.image_threshold  # pylint: disable=no-member
        meta["pixel_threshold"] = self.pixel_threshold  # pylint: disable=no-member
        meta["min"] = self.min  # pylint: disable=no-member
        meta["max"] = self.max  # pylint: disable=no-member

        anomaly_map = ((anomaly_map - meta["pixel_threshold"]) / (meta["max"] - meta["min"])) + 0.5
        pred_score = ((pred_score - meta["image_threshold"]) / (meta["max"] - meta["min"])) + 0.5

        input_image_height = meta["original_shape"][0]
        input_image_width = meta["original_shape"][1]
        result = cv2.resize(anomaly_map, (input_image_width, input_image_height))
        return result
