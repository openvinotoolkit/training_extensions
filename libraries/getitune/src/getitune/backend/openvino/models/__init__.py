# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO models implementation for all supported CV tasks."""

from .base import OVModel
from .detection import OVDetectionModel
from .instance_segmentation import OVInstanceSegmentationModel
from .keypoint_detection import OVKeypointDetectionModel
from .multiclass_classification import OVMulticlassClassificationModel
from .multilabel_classification import OVMultilabelClassificationModel
from .segmentation import OVSegmentationModel

__all__ = [
    "OVDetectionModel",
    "OVInstanceSegmentationModel",
    "OVKeypointDetectionModel",
    "OVModel",
    "OVMulticlassClassificationModel",
    "OVMultilabelClassificationModel",
    "OVSegmentationModel",
]
