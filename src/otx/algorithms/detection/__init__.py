"""OTX Algorithms - Detection."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

MMDET_AVAILABLE = True

try:
    import mmdet  # noqa: F401
except ImportError:
    MMDET_AVAILABLE = False
