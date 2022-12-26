# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdet.integration.nncf.utils import no_nncf_trace
from mmdet.models.detectors import BaseDetector, TwoStageDetector
from mmdet.utils.deployment.export_helpers import get_feature_vector

from otx.mpa.modules.hooks.recording_forward_hooks import DetSaliencyMapHook


class SAMDetectorMixin(BaseDetector):
    """SAM-enabled detector mix-in"""

    def train_step(self, data, optimizer, **kwargs):
        # Saving current batch data to compute SAM gradient
        # Rest of SAM logics are implented in SAMOptimizerHook
        self.current_batch = data
        return super().train_step(data, optimizer, **kwargs)

    def simple_test(self, img, img_metas, proposals=None, rescale=False, postprocess=True):
        """
        Class-wise Saliency map for Single-Stage Detector, otherwise use class-ignore saliency map.
        """
        if isinstance(self, TwoStageDetector):
            return super().simple_test(img, img_metas, proposals, rescale, postprocess)
        else:
            x = self.extract_feat(img)
            outs = self.bbox_head(x)
            with no_nncf_trace():
                bbox_results = self.bbox_head.get_bboxes(*outs, img_metas, self.test_cfg, False)
                if torch.onnx.is_in_onnx_export():
                    feature_vector = get_feature_vector(x)
                    cls_scores = outs[0]
                    saliency_map = DetSaliencyMapHook(self).func(cls_scores, cls_scores_provided=True)
                    feature = feature_vector, saliency_map
                    return bbox_results[0], feature

            if postprocess:
                bbox_results = [
                    self.postprocess(det_bboxes, det_labels, None, img_metas, rescale=rescale)
                    for det_bboxes, det_labels in bbox_results
                ]
            return bbox_results
