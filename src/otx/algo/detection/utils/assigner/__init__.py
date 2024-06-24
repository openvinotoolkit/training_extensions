# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Assigners for detection task."""

from .atss_assigner import ATSSAssigner
from .dynamic_soft_label_assigner import DynamicSoftLabelAssigner
from .max_iou_assigner import MaxIoUAssigner
from .sim_ota_assigner import SimOTAAssigner

__all__ = ["ATSSAssigner", "DynamicSoftLabelAssigner", "MaxIoUAssigner", "SimOTAAssigner"]
