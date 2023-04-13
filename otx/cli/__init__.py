"""OTX CLI."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

# How to make an Action Template invisible in Geti
# Check FEATURE_FLAGS_OTX_ACTION_TASKS in the API to determine whether to use the Action
# Always 1 in the OTX CLI
os.environ["FEATURE_FLAGS_OTX_ACTION_TASKS"] = "1"
