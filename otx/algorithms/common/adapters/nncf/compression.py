"""NNCF utils."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

from .utils import check_nncf_is_enabled, get_nncf_version


@dataclass
class NNCFMetaState:
    """NNCF meta state wrapper."""

    state_to_build: Optional[Dict[str, torch.Tensor]] = field(default=None)
    data_to_build: Optional[np.ndarray] = field(default=None)
    compression_ctrl: Optional[Dict[Any, Any]] = field(default=None)

    def __repr__(self):
        """Repr."""
        out = f"{self.__class__.__name__}("
        if self.state_to_build is not None:
            out += "state_to_build='<data>', "
        if self.data_to_build is not None:
            out += "data_to_build='<data>', "
        if self.compression_ctrl is not None:
            out += "compression_ctrl='<data>', "
        if out[-2:] == ", ":
            out = out[:-2]
        out += ")"
        return out


def get_nncf_metadata():
    """Get NNCF related metadata.

    The function returns NNCF metadata that should be stored into a checkpoint.
    The metadata is used to check in wrap_nncf_model if the checkpoint should be used
    to resume NNCF training or initialize NNCF fields of NNCF-wrapped model.
    """
    check_nncf_is_enabled()
    return dict(nncf_enable_compression=True, nncf_version=get_nncf_version())


def is_state_nncf(state):
    """Check if state_dict is NNCF state_dict.

    The function uses metadata stored in a dict_state to check if the
    checkpoint was the result of trainning of NNCF-compressed model.
    See the function get_nncf_metadata above.
    """
    return bool(state.get("meta", {}).get("nncf_enable_compression", False))


def is_checkpoint_nncf(path):
    """Check if path is NNCF checkpoint.

    The function uses metadata stored in a checkpoint to check if the
    checkpoint was the result of trainning of NNCF-compressed model.
    See the function get_nncf_metadata above.
    """
    try:
        checkpoint = torch.load(path, map_location="cpu")
        return is_state_nncf(checkpoint)
    except FileNotFoundError:
        return False


class AccuracyAwareLrUpdater:
    """AccuracyAwareLrUpdater."""

    def __init__(self, lr_hook):
        self._lr_hook = lr_hook
        self._lr_hook.warmup_iters = 0

    def step(self, *args, **kwargs):
        """step."""

    @property
    def base_lrs(self):
        """base_lrs."""
        return self._lr_hook.base_lr

    @base_lrs.setter
    def base_lrs(self, value):
        self._lr_hook.base_lr = value
