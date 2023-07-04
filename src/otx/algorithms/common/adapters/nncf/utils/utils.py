"""NNCF utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from collections import OrderedDict
from contextlib import contextmanager
from importlib.util import find_spec

import torch

_is_nncf_enabled = find_spec("nncf") is not None


def is_nncf_enabled():
    """is_nncf_enabled."""
    return _is_nncf_enabled


def check_nncf_is_enabled():
    """check_nncf_is_enabled."""
    if not is_nncf_enabled():
        raise RuntimeError("Tried to use NNCF, but NNCF is not installed")


def get_nncf_version():
    """get_nncf_version."""
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
    #  from nncf.torch import load_state
    from mmcv.runner import load_state_dict

    checkpoint = torch.load(filename, map_location=map_location)
    nncf_state = None
    compression_state = None
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        base_state = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        if "meta" in checkpoint and "nncf_meta" in checkpoint["meta"]:
            nncf_state = checkpoint["state_dict"]
            compression_state = checkpoint["meta"]["nncf_meta"].compression_ctrl
            base_state = checkpoint["meta"]["nncf_meta"].state_to_build
        else:
            base_state = checkpoint["state_dict"]
    else:
        raise RuntimeError(f"No state_dict found in checkpoint file {filename}")
    load_state_dict(model, base_state, strict=strict)
    return compression_state, nncf_state


@contextmanager
def nullcontext():
    """Context which does nothing."""
    yield


def no_nncf_trace():
    """Wrapper for original NNCF no_nncf_trace context."""

    if is_nncf_enabled():
        from nncf.torch.dynamic_graph.context import (
            no_nncf_trace as original_no_nncf_trace,
        )

        return original_no_nncf_trace()
    return nullcontext()


def nncf_trace():
    """Trace nncf context."""
    if is_nncf_enabled():

        @contextmanager
        def _nncf_trace():
            from nncf.torch.dynamic_graph.context import get_current_context

            ctx = get_current_context()
            if ctx is not None and not ctx.is_tracing:
                ctx.enable_tracing()
                yield
                ctx.disable_tracing()
            else:
                yield

        return _nncf_trace()
    return nullcontext()


def is_in_nncf_tracing():
    """is_in_nncf_tracing."""
    if not is_nncf_enabled():
        return False

    from nncf.torch.dynamic_graph.context import get_current_context

    ctx = get_current_context()

    if ctx is None:
        return False
    return ctx.is_tracing


def is_accuracy_aware_training_set(nncf_config):
    """is_accuracy_aware_training_set."""
    if not is_nncf_enabled():
        return False
    from nncf.config.utils import is_accuracy_aware_training

    is_acc_aware_training_set = is_accuracy_aware_training(nncf_config)
    return is_acc_aware_training_set
