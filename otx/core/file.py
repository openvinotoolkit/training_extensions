"""File system related utilities."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

OTX_CACHE = os.path.expanduser(
    os.getenv(
        "OTX_CACHE",
        os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "otx"),
    )
)
os.makedirs(OTX_CACHE, exist_ok=True)
