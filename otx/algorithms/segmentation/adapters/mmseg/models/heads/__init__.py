"""Semantic segmentation heads."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .custom_fcn_head import CustomFCNHead
from .mmov_decode_head import MMOVDecodeHead

__all__ = ["MMOVDecodeHead", "CustomFCNHead"]
