"""Custom Deformable DETR head for OTX."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import HEADS
from mmdet.models.dense_heads.deformable_detr_head import DeformableDETRHead


@HEADS.register_module()
class CustomDeformableDETRHead(DeformableDETRHead):
    """Custom Deformable DETR Head.

    Since batch_input_shape are not added in mmdeploy, here this function add it.
    However additional if condition may leads time consumption therefore we need to
    find better way to add "batch_input_shape" to img_metas when the model is exported.
    """

    def forward(self, mlvl_feats, img_metas):
        """Modified forward function for onnx export."""

        if "batch_input_shape" not in img_metas[0]:
            height = int(img_metas[0]["img_shape"][0])
            width = int(img_metas[0]["img_shape"][1])
            img_metas[0]["batch_input_shape"] = (height, width)
            img_metas[0]["img_shape"] = (height, width, 3)
        return super().forward(mlvl_feats, img_metas)
