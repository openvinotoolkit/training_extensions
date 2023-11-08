"""MoViNet Recognizer for OTX compatibility."""
# pylint: disable=unused-argument
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools

from mmaction.models.recognizers.recognizer3d import Recognizer3D
from mmaction.registry import MODELS


@MODELS.register_module()
class MoViNetRecognizer(Recognizer3D):
    """MoViNet recognizer model framework for OTX compatibility."""

    def __init__(self, **kwargs) -> None:
        """Initialization."""
        super().__init__(**kwargs)
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    @staticmethod
    def state_dict_hook(state_dict: dict) -> None:
        """Redirect model as output state_dict for OTX MoviNet compatibility."""
        new_ckpt = {}
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            new_key = key.replace("cls_head.", "") if "cls_head" in key else key.replace("backbone.", "")
            new_ckpt[new_key] = val
        state_dict = new_ckpt

    @staticmethod
    def load_state_dict_pre_hook(state_dict: dict, prefix:str) -> None:
        """Redirect input state_dict to model for OTX model compatibility."""
        new_ckpt = {}
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            clf = "classifier"
            new_key = key.replace(clf, f"cls_head.{clf}") if "clf" in key else prefix + "backbone." + key[len(prefix) :]
            new_ckpt[new_key] = val
        state_dict = new_ckpt
