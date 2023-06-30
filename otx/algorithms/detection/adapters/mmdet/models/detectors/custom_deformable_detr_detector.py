"""OTX Deformable DETR Class for mmdetection detectors."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.deformable_detr import DeformableDETR

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled


@DETECTORS.register_module()
class CustomDeformableDETR(DeformableDETR):
    """Custom Deformable DETR with task adapt.

    Deformable DETR does not support task adapt, so it just take task_adpat argument.
    """

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_adapt = task_adapt


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_deformable_detr_detector.CustomDeformableDETR.simple_test"
    )
    def custom_deformable_detr__simple_test(ctx, self, img, img_metas, **kwargs):
        """Function for custom_deformable_detr__simple_test."""
        height = int(img_metas[0]["img_shape"][0])
        width = int(img_metas[0]["img_shape"][1])
        img_metas[0]["batch_input_shape"] = (height, width)
        img_metas[0]["img_shape"] = (height, width, 3)
        feat = self.extract_feat(img)
        outs = self.bbox_head(feat, img_metas)
        bbox_results = self.bbox_head.get_bboxes(*outs, img_metas=img_metas, **kwargs)

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(feat)
            cls_scores = outs[0]
            saliency_map = ActivationMapHook.func(cls_scores)
            return (*bbox_results, feature_vector, saliency_map)

        return bbox_results
