"""OTX Algorithms - Action Recognition."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["FEATURE_FLAGS_OTX_ACTION_TASKS"] = "1"

MMACTION_AVAILABLE = True

try:
    import mmaction  # noqa: F401
except ImportError:
    MMACTION_AVAILABLE = False
