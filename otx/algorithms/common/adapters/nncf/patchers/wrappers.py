# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.nncf.utils import (
    nncf_trace,
    no_nncf_trace,
)


def no_nncf_trace_wrapper(self, fn, *args, **kwargs):
    """
    A wrapper function not to trace in NNCF.
    """

    with no_nncf_trace():
        return fn(*args, **kwargs)


def nncf_trace_wrapper(self, fn, *args, **kwargs):
    """
    A wrapper function to trace in NNCF.
    """

    with nncf_trace():
        return fn(*args, **kwargs)
