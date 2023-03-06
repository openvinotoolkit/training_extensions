# pylint: disable-all
# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import functools
from collections import OrderedDict

from mmaction.models.builder import RECOGNIZERS
from mmaction.models.recognizers.recognizer3d import Recognizer3D


@RECOGNIZERS.register_module(force=True)
class OTXRecognizer3D(Recognizer3D):
    """3D recognizer model framework for OTX compatibility"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Hooks for redirect state_dict load/save
        self._register_state_dict_hook(self.state_dict_hook)
        self._register_load_state_dict_pre_hook(functools.partial(self.load_state_dict_pre_hook, self))

    @staticmethod
    def state_dict_hook(module, state_dict, *args, **kwargs):
        """Redirect model as output state_dict for OTX model compatibility"""
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["OTXMoViNet"]:
            return

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if "cls_head" in k:
                k = k.replace("cls_head.", "")
            else:
                k = k.replace("backbone.", "")
            state_dict[k] = v

    @staticmethod
    def load_state_dict_pre_hook(module, state_dict, prefix, *args, **kwargs):
        """Redirect input state_dict to model for OTX model compatibility"""
        backbone_type = type(module.backbone).__name__
        if backbone_type not in ["OTXMoViNet"]:
            return

        for k in list(state_dict.keys()):
            v = state_dict.pop(k)
            if "classifier" in k:
                k = k.replace("classifier", "cls_head.classifier")
            else:
                k = prefix + "backbone." + k[len(prefix):]
            state_dict[k] = v
