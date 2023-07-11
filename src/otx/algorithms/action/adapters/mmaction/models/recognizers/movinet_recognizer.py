"""MoViNet Recognizer for OTX compatibility."""
# pylint: disable=unused-argument
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools

from mmaction.models.builder import RECOGNIZERS
from mmaction.models.recognizers.recognizer3d import Recognizer3D


@RECOGNIZERS.register_module()
class MoViNetRecognizer(Recognizer3D):
    """MoViNet recognizer model framework for OTX compatibility."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect model as output state_dict for OTX MoviNet compatibility."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if "cls_head" in key:
                key = key.replace("cls_head.", "")
            else:
                key = key.replace("backbone.", "")
            state_dict[key] = val

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, prefix, *args, **kwargs):
        """Redirect input state_dict to model for OTX model compatibility."""
        for key in list(state_dict.keys()):
            val = state_dict.pop(key)
            if "classifier" in key:
                key = key.replace("classifier", "cls_head.classifier")
            else:
                key = prefix + "backbone." + key[len(prefix) :]
            state_dict[key] = val
