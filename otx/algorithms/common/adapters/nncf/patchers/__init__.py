"""simple monky patch helper."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .patcher import Patcher
from .wrappers import nncf_trace_wrapper, no_nncf_trace_wrapper

NNCF_PATCHER = Patcher()


__all__ = [
    "Patcher",
    "NNCF_PATCHER",
    "nncf_trace_wrapper",
    "no_nncf_trace_wrapper",
]
