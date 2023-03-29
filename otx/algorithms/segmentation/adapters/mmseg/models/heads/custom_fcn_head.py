"""Custom FCN head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.decode_heads.sep_aspp_head import ASPPHead

from .mixin import (
    AggregatorMixin,
    MixLossMixin,
    PixelWeightsMixin2,
    SegmentOutNormMixin,
)

KNOWN_HEADS = {
    "FCNHead": FCNHead,
    "ASPPHead": ASPPHead
}

def get_head(head_name, *args, **kwargs):
    head_class = KNOWN_HEADS[head_name]

    class CustomOTXHead(SegmentOutNormMixin, AggregatorMixin, MixLossMixin, PixelWeightsMixin2, head_class):
        """Custom Fully Convolution Networks for Semantic Segmentation."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if kwargs.get("init_cfg", {}):
                self.init_weights()

    return CustomOTXHead(*args, **kwargs)



@HEADS.register_module()
class CustomFCNHead(
    SegmentOutNormMixin, AggregatorMixin, MixLossMixin, PixelWeightsMixin2, FCNHead
):  # pylint: disable=too-many-ancestors
    """Custom Fully Convolution Networks for Semantic Segmentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # get rid of last activation of convs module
        if self.act_cfg:
            self.convs[-1].with_activation = False
            delattr(self.convs[-1], "activate")

        if kwargs.get("init_cfg", {}):
            self.init_weights()
