"""Backbone Class of OMZ model for mmdetection backbones."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import BACKBONES

from otx.core.ov.models.mmov_model import MMOVModel


@BACKBONES.register_module()
class MMOVBackbone(MMOVModel):
    """MMOVBackbone Class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward function of MMOVBackbone."""
        outputs = super().forward(*args, **kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # must return tuple
        return outputs

    def init_weights(self, pretrained=None):  # pylint: disable=unused-argument
        """Initial weights function of MMOVBackbone."""
        # TODO
        return
