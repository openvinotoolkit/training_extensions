"""NNCFNetwork patch util functions for mmcv models."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from contextlib import contextmanager
from functools import partial

from otx.algorithms.common.adapters.nncf.utils import nncf_trace, no_nncf_trace


@contextmanager
def nncf_trace_context(self, img_metas, nncf_compress_postprocessing=True):
    """A context manager for nncf graph tracing."""

    # onnx_export in mmdet head has a bug on GPU
    # it must be on CPU
    device_backup = next(self.parameters()).device  # pylint: disable=stop-iteration-return
    self = self.to("cpu")

    if nncf_compress_postprocessing:
        self.forward = partial(self.forward, img_metas=img_metas, return_loss=False)
    else:
        self.forward = partial(self.forward_dummy)

    yield

    # make everything normal
    self.__dict__.pop("forward")
    self = self.to(device_backup)


def no_nncf_trace_wrapper(self, fn, *args, **kwargs):  # pylint: disable=unused-argument,invalid-name
    """A wrapper function not to trace in NNCF."""

    with no_nncf_trace():
        return fn(*args, **kwargs)


def nncf_trace_wrapper(self, fn, *args, **kwargs):  # pylint: disable=unused-argument,invalid-name
    """A wrapper function to trace in NNCF."""

    with nncf_trace():
        return fn(*args, **kwargs)
