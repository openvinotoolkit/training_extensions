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
from openvino.model_zoo.model_api.models.types import ListValue, NumericalValue
from scipy.stats import norm


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
                "pixel_threshold": NumericalValue(description="Pixel Threshold value to locate anomaly"),
                "pixel_mean": ListValue(description="Pixel mean used to normalize"),
                "pixel_std": ListValue(description="Pixel standard deviation value used to normalize"),
                "image_mean": NumericalValue(description="Image mean value used to normalize"),
                "image_std": NumericalValue(description="Image standard deviation used to normalize"),
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
        anomaly_map = outputs[self.output_blob_name].squeeze()
        pred_score = anomaly_map.reshape(-1).max()

        meta["image_threshold"] = self.image_threshold  # pylint: disable=no-member
        meta["pixel_mean"] = np.array(self.pixel_mean)  # pylint: disable=no-member
        meta["pixel_std"] = np.array(self.pixel_std)  # pylint: disable=no-member
        meta["pixel_threshold"] = self.pixel_threshold  # pylint: disable=no-member
        meta["image_mean"] = self.image_mean  # pylint: disable=no-member
        meta["image_std"] = self.image_std  # pylint: disable=no-member

        # standardize pixel scores
        if "pixel_mean" in meta.keys() and "pixel_std" in meta.keys():
            anomaly_map = np.log(anomaly_map)
            anomaly_map = (anomaly_map - meta["pixel_mean"]) / meta["pixel_std"]
            anomaly_map -= (meta["image_mean"] - meta["pixel_mean"]) / meta["pixel_std"]
            anomaly_map = norm.cdf(anomaly_map - meta["pixel_threshold"])

        # standardize image scores
        if "image_mean" in meta.keys() and "image_std" in meta.keys():
            pred_score = np.log(pred_score)
            pred_score = (pred_score - meta["image_mean"]) / meta["image_std"]
            pred_score = norm.cdf(pred_score - meta["image_threshold"])

        input_image_height = meta["original_shape"][0]
        input_image_width = meta["original_shape"][1]

        result = cv2.resize(anomaly_map, (input_image_width, input_image_height))
        return result
