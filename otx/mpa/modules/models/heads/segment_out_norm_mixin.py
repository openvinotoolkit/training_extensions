# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch.nn as nn

from ..utils import AngularPWConv, normalize


class SegmentOutNormMixin(nn.Module):
    def __init__(self, *args, enable_out_seg=True, enable_out_norm=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.enable_out_seg = enable_out_seg
        self.enable_out_norm = enable_out_norm

        if enable_out_seg:
            if enable_out_norm:
                self.conv_seg = AngularPWConv(self.channels, self.out_channels, clip_output=True)
        else:
            self.conv_seg = None

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        if self.enable_out_norm:
            feat = normalize(feat, dim=1, p=2)
        if self.conv_seg is not None:
            return self.conv_seg(feat)
        else:
            return feat
