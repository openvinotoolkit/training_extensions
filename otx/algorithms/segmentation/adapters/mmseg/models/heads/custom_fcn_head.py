"""Custom FCN head."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmcv.runner import force_fp32
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.fcn_head import FCNHead

from otx.algorithms.segmentation.adapters.mmseg.models.utils import LossEqualizer


@HEADS.register_module()
class CustomFCNHead(FCNHead):  # pylint: disable=too-many-ancestors
    """Custom Fully Convolution Networks for Semantic Segmentation."""

    def __init__(self, enable_loss_equalizer=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_equalizer = False
        if enable_loss_equalizer:
            self.loss_equalizer = LossEqualizer()

    @force_fp32(apply_to=("seg_logit",))
    def losses(self, seg_logit, seg_label):
        loss = super().losses(seg_logit, seg_label)
        if self.loss_equalizer:
            out_loss = self.loss_equalizer.reweight(loss)
            for loss_name, loss_value in out_loss.items():
                loss[loss_name] = loss_value
        return loss
