# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""MoViNet Recognizer for OTX compatibility."""

import functools

from mmaction.models import MODELS
from mmaction.models.recognizers.recognizer3d import Recognizer3D
from torch import nn


@MODELS.register_module()
class MoViNetRecognizer(Recognizer3D):
    """MoViNet recognizer model framework for OTX compatibility."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    @staticmethod
    def state_dict_hook(module: nn.Module, state_dict: dict, *args, **kwargs) -> None:  # noqa: ARG004
        """Redirect model as output state_dict for OTX MoviNet compatibility."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            key = key.replace("cls_head.", "") if "cls_head" in key else key.replace("backbone.", "")  # noqa: PLW2901
            state_dict[key] = val

    @staticmethod
    def load_state_dict_pre_hook(
        module: nn.Module,  # noqa: ARG004
        state_dict: dict,
        prefix: str,
        *args,  # noqa: ARG004
        **kwargs,  # noqa: ARG004
    ) -> None:
        """Redirect input state_dict to model for OTX model compatibility."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            key = (  # noqa: PLW2901
                key.replace("classifier", "cls_head.classifier")
                if "classifier" in key
                else prefix + "backbone." + key[len(prefix) :]
            )
            state_dict[key] = val
