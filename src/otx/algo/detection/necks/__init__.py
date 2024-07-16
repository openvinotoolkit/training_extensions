# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Custom neck implementations for detection task."""

from .cspnext_pafpn import CSPNeXtPAFPN
from .fpn import FPN
from .hybrid_encoder import HybridEncoder
from .yolox_pafpn import YOLOXPAFPN

__all__ = ["CSPNeXtPAFPN", "FPN", "YOLOXPAFPN", "HybridEncoder"]
