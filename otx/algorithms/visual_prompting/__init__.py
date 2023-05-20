"""OTX Algorithms - Action Recognition."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["FEATURE_FLAGS_OTX_VISUAL_PROMPTING_TASKS"] = "1"

VISUAL_PROMPTING_AVAILABLE = True

try:
    import segment_anything  # noqa: F401
except ImportError:
    VISUAL_PROMPTING_AVAILABLE = False
