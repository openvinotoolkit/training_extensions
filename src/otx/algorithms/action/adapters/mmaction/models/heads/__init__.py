"""OTX Adapters for action recognition backbones - mmaction2."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .movinet_head import MoViNetHead
from .roi_head import AVARoIHead

__all__ = ["AVARoIHead", "MoViNetHead"]
