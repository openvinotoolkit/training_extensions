"""Custom universal class incremental otx head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmseg.models.decode_heads.fcn_head import FCNHead
from mmseg.models.decode_heads.sep_aspp_head import DepthwiseSeparableASPPHead

from .light_ham import LightHamHead
from .mixin import SegMixinModule

KNOWN_HEADS = {"FCNHead": FCNHead, "ASPPHead": DepthwiseSeparableASPPHead, "LightHamHead": LightHamHead}


def otx_head_factory(*args, base_type="FCNHead", **kwargs):
    """Factory function for creating custom otx head based on mmsegmentation heads."""

    head_base_cls = KNOWN_HEADS[base_type]

    class CustomOTXHead(SegMixinModule, head_base_cls):
        """Custom universal class incremental head for Semantic Segmentation."""

        def __init__(self, head_type, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # get rid of last activation of convs module
            if head_type == "FCNHead" and self.act_cfg:
                self.convs[-1].with_activation = False
                delattr(self.convs[-1], "activate")

            if kwargs.get("init_cfg", {}):
                self.init_weights()

    return CustomOTXHead(base_type, *args, **kwargs)
