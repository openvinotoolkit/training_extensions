# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .patcher import Patcher
from .wrappers import nncf_trace_wrapper, no_nncf_trace_wrapper


NO_TRACE_PATCHER = Patcher(no_nncf_trace_wrapper)
TRACE_PATCHER = Patcher(nncf_trace_wrapper)


__all__ = [
    "Patcher",
    "NO_TRACE_PATCHER",
    "TRACE_PATCHER",
    "nncf_trace_wrapper",
    "no_nncf_trace_wrapper",
]
