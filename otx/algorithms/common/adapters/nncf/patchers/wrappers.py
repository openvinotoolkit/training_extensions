# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.algorithms.common.adapters.nncf.utils import (
    nncf_trace,
    no_nncf_trace,
)


def no_nncf_trace_wrapper(*args, **kwargs):
    """
    A wrapper function not to trace in NNCF.
    """

    in_fn = kwargs.pop("in_fn")
    with no_nncf_trace():
        return in_fn(*args, **kwargs)


def nncf_trace_wrapper(*args, **kwargs):
    """
    A wrapper function to trace in NNCF.
    """

    in_fn = kwargs.pop("in_fn")
    with nncf_trace():
        return in_fn(*args, **kwargs)
