# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""getitune task type definition."""

from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """getitune task type definition."""

    # Classification
    MULTI_CLASS_CLS = "MULTI_CLASS_CLS"
    MULTI_LABEL_CLS = "MULTI_LABEL_CLS"

    # Detection
    DETECTION = "DETECTION"
    KEYPOINT_DETECTION = "KEYPOINT_DETECTION"

    # Segmentation
    INSTANCE_SEGMENTATION = "INSTANCE_SEGMENTATION"
    SEMANTIC_SEGMENTATION = "SEMANTIC_SEGMENTATION"
