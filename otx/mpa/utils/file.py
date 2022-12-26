# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

MPA_CACHE = os.path.expanduser(os.getenv("MPA_CACHE", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "mpa")))
os.makedirs(MPA_CACHE, exist_ok=True)
