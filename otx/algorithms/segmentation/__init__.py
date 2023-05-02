"""OTX Algorithms - Segmentation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

MMSEG_AVAILABLE = True

try:
    import mmseg  # noqa: F401
except ImportError:
    MMSEG_AVAILABLE = False
