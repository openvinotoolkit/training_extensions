# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX task type definition."""

from __future__ import annotations

from enum import Enum


class OTXTaskType(str, Enum):
    """OTX task type definition."""

    MULTI_CLASS_CLS = "MULTI_CLASS_CLS"
    DETECTION = "DETECTION"
    INSTANCE_SEGMENTATION = "INSTANCE_SEGMENTATION"
    DETECTION_SEMI_SL = "DETECTION_SEMI_SL"
    SEMANTIC_SEGMENTATION = "SEMANTIC_SEGMENTATION"
    ACTION_CLASSIFICATION = "ACTION_CLASSIFICATION"
