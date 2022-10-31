# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from .compression import (
    get_nncf_config_from_meta,
    wrap_nncf_model,
)

from .monkey_patch import *

__all__ = [
    "get_nncf_config_from_meta",
    "wrap_nncf_model",
]
