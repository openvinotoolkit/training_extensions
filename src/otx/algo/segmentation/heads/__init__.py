# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Head modules for OTX segmentation model."""

from .fcn_head import FCNHead
from .ham_head import LightHamHead

__all__ = ["FCNHead", "LightHamHead"]
