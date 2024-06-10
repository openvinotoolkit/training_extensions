# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MoViNet Recognizer for OTX compatibility."""
import functools

from torch import nn

from otx.algo.action_classification.recognizers.recognizer import BaseRecognizer


class MoViNetRecognizer(BaseRecognizer):
    """MoViNet recognizer model framework for OTX compatibility."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    @staticmethod
    def state_dict_hook(module: nn.Module, state_dict: dict, *args, **kwargs) -> None:  # noqa: ARG004
        """Redirect model as output state_dict for OTX MoviNet compatibility."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            new_key = key.replace("cls_head.", "") if "cls_head" in key else key.replace("backbone.", "")
            state_dict[new_key] = val

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
            new_key = (
                key.replace("classifier", "cls_head.classifier")
                if "classifier" in key
                else prefix + "backbone." + key[len(prefix) :]
            )
            state_dict[new_key] = val
