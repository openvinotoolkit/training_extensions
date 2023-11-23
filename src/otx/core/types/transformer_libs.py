"""Transform library types used in OTX."""

from __future__ import annotations

from enum import Enum


class TransformLibType(str, Enum):
    """Transform library types used in OTX."""

    TORCHVISION = "TORCHVISION"
    MMCV = "MMCV"
    MMDET = "MMDET"
