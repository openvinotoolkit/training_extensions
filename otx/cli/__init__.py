"""OTX CLI."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

from otx import MMACTION_AVAILABLE

if MMACTION_AVAILABLE:
    os.environ["FEATURE_FLAGS_OTX_ACTION_TASKS"] = "1"
