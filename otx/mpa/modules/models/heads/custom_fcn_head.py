# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead

from .aggregator_mixin import AggregatorMixin
from .mix_loss_mixin import MixLossMixin
from .pixel_weights_mixin import PixelWeightsMixin2
from .segment_out_norm_mixin import SegmentOutNormMixin


@HEADS.register_module()
class CustomFCNHead(SegmentOutNormMixin, AggregatorMixin, MixLossMixin, PixelWeightsMixin2, FCNHead):
    """Custom Fully Convolution Networks for Semantic Segmentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get rid of last activation of convs module
        if self.act_cfg:
            self.convs[-1].with_activation = False
            delattr(self.convs[-1], "activate")

        if kwargs.get("init_cfg", {}):
            self.init_weights()
