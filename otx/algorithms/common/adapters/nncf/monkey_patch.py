# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from contextlib import contextmanager
from functools import partial, partialmethod

import torch

from otx.algorithms.common.adapters.nncf.utils import (
    is_nncf_enabled,
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


def conditioned_wrapper(target_cls, wrapper, methods, subclasses=(object,)):
    """
    A function to wrap all the given methods under given subclasses.
    """

    if issubclass(target_cls, subclasses):
        for func_name in methods:
            func = getattr(target_cls, func_name, None)
            if func is not None and "_partialmethod" not in func.__dict__:
                #  print(target_cls, func_name)
                setattr(target_cls, func_name, partialmethod(wrapper, in_fn=func))


@contextmanager
def nncf_trace_context(self, img_metas):
    """
    A context manager for nncf graph tracing
    """

    if is_nncf_enabled():
        # onnx_export in mmdet head has a bug on GPU
        # it must be on CPU
        device_backup = next(self.parameters()).device
        self = self.to("cpu")
        # HACK
        # temporarily change current context as onnx export context
        # to trace network
        onnx_backup = torch.onnx.utils.__IN_ONNX_EXPORT
        torch.onnx.utils.__IN_ONNX_EXPORT = True
        # backup forward
        forward_backup = self.forward
        self.forward = partial(self.forward, img_metas=img_metas)

    yield

    if is_nncf_enabled():
        # make everything normal
        self.forward = forward_backup
        torch.onnx.utils.__IN_ONNX_EXPORT = onnx_backup
        self = self.to(device_backup)
