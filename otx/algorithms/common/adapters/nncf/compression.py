# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch

from .utils import check_nncf_is_enabled, get_nncf_version, is_nncf_enabled


NNCF_STATE_NAME = "nncf_model_state"
COMPRESSION_STATE_NAME = "compression_state"
DATA_TO_BUILD_NAME = "data_to_build_nncf"
STATE_TO_BUILD_NAME = "state_dict_to_build_nncf"


def get_nncf_metadata():
    """
    The function returns NNCF metadata that should be stored into a checkpoint.
    The metadata is used to check in wrap_nncf_model if the checkpoint should be used
    to resume NNCF training or initialize NNCF fields of NNCF-wrapped model.
    """
    check_nncf_is_enabled()
    return dict(nncf_enable_compression=True, nncf_version=get_nncf_version())


def is_state_nncf(state):
    """
    The function uses metadata stored in a dict_state to check if the
    checkpoint was the result of trainning of NNCF-compressed model.
    See the function get_nncf_metadata above.
    """
    return bool(state.get("meta", {}).get("nncf_enable_compression", False))


def is_checkpoint_nncf(path):
    """
    The function uses metadata stored in a checkpoint to check if the
    checkpoint was the result of trainning of NNCF-compressed model.
    See the function get_nncf_metadata above.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu")
        return is_state_nncf(checkpoint)
    except FileNotFoundError:
        return False


def get_uncompressed_model(module):
    if not is_nncf_enabled():
        return module
    from nncf.torch.nncf_network import NNCFNetwork

    if isinstance(module, NNCFNetwork):
        return module.get_nncf_wrapped_model()
    return module


class AccuracyAwareLrUpdater:
    def __init__(self, lr_hook):
        self._lr_hook = lr_hook
        self._lr_hook.warmup_iters = 0

    def step(self, *args, **kwargs):
        pass

    @property
    def base_lrs(self):
        return self._lr_hook.base_lr

    @base_lrs.setter
    def base_lrs(self, value):
        self._lr_hook.base_lr = value
