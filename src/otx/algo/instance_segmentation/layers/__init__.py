# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom layer implementations for instance segmentation task."""

from .bbox_nms import multiclass_nms_torch

__all__ = ["multiclass_nms_torch"]
