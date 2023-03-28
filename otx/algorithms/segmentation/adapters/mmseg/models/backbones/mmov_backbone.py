"""Backbone used for openvino export."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.models.builder import BACKBONES

from otx.core.ov.models.mmov_model import MMOVModel

# pylint: disable=unused-argument


@BACKBONES.register_module()
class MMOVBackbone(MMOVModel):
    """MMOVBackbone."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Forward."""
        outputs = super().forward(*args, **kwargs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # must return tuple
        return outputs

    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        # TODO
        return
