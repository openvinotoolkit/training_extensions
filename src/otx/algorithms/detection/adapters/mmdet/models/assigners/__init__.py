"""Initial file for mmdetection assigners."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .custom_max_iou_assigner import CustomMaxIoUAssigner
from .xpu_atss_assigner import XPUATSSAssigner
from .custom_sim_ota_assigner import CustomSimOTAAssigner

__all__ = ["CustomMaxIoUAssigner", "XPUATSSAssigner", "CustomSimOTAAssigner"]
