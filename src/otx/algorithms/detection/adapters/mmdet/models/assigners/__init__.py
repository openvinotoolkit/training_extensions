"""Initial file for mmdetection assigners."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .custom_max_iou_assigner import CustomMaxIoUAssigner

__all__ = ["CustomMaxIoUAssigner"]
