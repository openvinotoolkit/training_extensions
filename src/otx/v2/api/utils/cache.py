"""File system related utilities."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from pathlib import Path

OTX_CACHE = Path(
    os.getenv(
        "OTX_CACHE",
        str(Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "otx"),
    ),
).expanduser()
OTX_CACHE.mkdir(parents=True, exist_ok=True)
