# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Head modules for OTX segmentation model."""

from .custom_fcn_head import CustomFCNHead
from .custom_ham_head import CustomLightHamHead

__all__ = ["CustomFCNHead", "CustomLightHamHead"]
