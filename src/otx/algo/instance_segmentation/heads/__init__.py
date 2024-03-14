# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Custom head architecture for OTX instance segmentation models."""

from .custom_roi_head import CustomConvFCBBoxHead, CustomRoIHead

__all__ = ["CustomRoIHead", "CustomConvFCBBoxHead"]
