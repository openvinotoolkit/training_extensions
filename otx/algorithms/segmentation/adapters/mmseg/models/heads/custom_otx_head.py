"""Custom universal class incremental otx head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead

from .mixin import Mixin

KNOWN_HEADS = {
    "FCNHead": FCNHead,
    "ASPPHead": DepthwiseSeparableASPPHead
}

def get_head(head_name, *args, **kwargs):
    head_class = KNOWN_HEADS[head_name]

    class CustomOTXHead(Mixin, head_class):
        """Custom universal class incremental head for Semantic Segmentation."""

        def __init__(self, head_name, *args, **kwargs):
            super().__init__(*args, **kwargs)

            if kwargs.get("init_cfg", {}):
                self.init_weights()

            # get rid of last activation of convs module
            if head_name == "FCNHead" and self.act_cfg:
                self.convs[-1].with_activation = False
                delattr(self.convs[-1], "activate")

    return CustomOTXHead(head_name, *args, **kwargs)
