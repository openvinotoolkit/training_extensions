# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
from collections import OrderedDict
from contextlib import contextmanager

import torch


_is_nncf_enabled = importlib.util.find_spec("nncf") is not None


def is_nncf_enabled():
    return _is_nncf_enabled


def check_nncf_is_enabled():
    if not is_nncf_enabled():
        raise RuntimeError("Tried to use NNCF, but NNCF is not installed")


def get_nncf_version():
    if not is_nncf_enabled():
        return None
    import nncf

    return nncf.__version__


def load_checkpoint(model, filename, map_location=None, strict=False):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Either a filepath or URL or modelzoo://xxxxxxx.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    from nncf.torch import load_state

    checkpoint = torch.load(filename, map_location=map_location)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise RuntimeError("No state_dict found in checkpoint file {}".format(filename))
    _ = load_state(model, state_dict, strict)
    return checkpoint


@contextmanager
def nullcontext():
    """
    Context which does nothing
    """
    yield


def no_nncf_trace():
    """
    Wrapper for original NNCF no_nncf_trace() context
    """

    if is_nncf_enabled():
        from nncf.torch.dynamic_graph.context import (
            no_nncf_trace as original_no_nncf_trace,
        )

        return original_no_nncf_trace()
    return nullcontext()


@contextmanager
def nncf_trace():
    from nncf.torch.dynamic_graph.context import get_current_context
    ctx = get_current_context()
    if ctx is not None and not ctx.is_tracing:
        ctx.enable_tracing()
        yield
        ctx.disable_tracing()
    else:
        yield


def is_in_nncf_tracing():
    if not is_nncf_enabled():
        return False

    from nncf.torch.dynamic_graph.context import get_current_context

    ctx = get_current_context()

    if ctx is None:
        return False
    return ctx.is_tracing


def is_accuracy_aware_training_set(nncf_config):
    if not is_nncf_enabled():
        return False
    from nncf.config.utils import is_accuracy_aware_training

    is_acc_aware_training_set = is_accuracy_aware_training(nncf_config)
    return is_acc_aware_training_set
