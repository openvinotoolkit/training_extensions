# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""utils for detection task."""

from .mmcv_patched_ops import monkey_patched_nms, monkey_patched_roi_align

__all__ = ["monkey_patched_nms", "monkey_patched_roi_align"]
