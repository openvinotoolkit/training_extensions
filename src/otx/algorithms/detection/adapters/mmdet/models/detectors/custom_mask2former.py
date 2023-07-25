"""OTX Mask2Former Class for mmdetection detectors."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.mask2former import Mask2Former

from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.utils.logger import get_logger

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()


@DETECTORS.register_module()
class CustomMask2Former(SAMDetectorMixin, L2SPDetectorMixin, Mask2Former):
    """CustomMask2Former Class for mmdetection detectors."""

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)


if is_mmdeploy_enabled():
    import torch
    from mmdeploy.core import FUNCTION_REWRITER

    from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
        FeatureVectorHook,
    )

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_mask2former.CustomMask2Former.simple_test"
    )
    def custom_mask2former__simple_test(ctx, self, img, img_metas, **kwargs):
        """Function for custom_mask2former__simple_test."""
        height = int(img_metas[0]["img_shape"][0])
        width = int(img_metas[0]["img_shape"][1])
        img_metas[0]["batch_input_shape"] = (height, width)
        img_metas[0]["img_shape"] = (height, width, 3)

        feats = self.extract_feat(img)
        mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(feats, img_metas, **kwargs)
        results = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)

        out = results[0]["ins_results"]

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(feats)
            # Saliency map will be generated from predictions. Generate dummy saliency_map.
            saliency_map = torch.empty(1, dtype=torch.uint8)
            return (*out, feature_vector, saliency_map)

        return out
