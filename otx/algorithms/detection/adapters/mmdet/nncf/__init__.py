# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .builder import (
    build_nncf_detector,
)

from .patches import *

__all__ = [
    "build_nncf_detector",
]
