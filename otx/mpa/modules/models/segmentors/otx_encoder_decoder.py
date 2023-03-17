# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmseg.models import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled


@SEGMENTORS.register_module()
class OTXEncoderDecoder(EncoderDecoder):
    def simple_test(self, img, img_meta, rescale=True, output_logits=False):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if output_logits:
            seg_pred = seg_logit
        else:
            if self.out_channels == 1:
                seg_pred = (seg_logit > self.decode_head.threshold).to(seg_logit).squeeze(1)
            else:
                seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            if seg_pred.dim() != 4:
                seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER

    from otx.mpa.modules.hooks.recording_forward_hooks import FeatureVectorHook

    @FUNCTION_REWRITER.register_rewriter(
        "otx.mpa.modules.models.segmentors.otx_encoder_decoder.OTXEncoderDecoder.extract_feat"
    )
    def single_stage_detector__extract_feat(ctx, self, img):
        feat = self.backbone(img)
        self.feature_map = feat
        if self.with_neck:
            feat = self.neck(feat)
        return feat

    @FUNCTION_REWRITER.register_rewriter(
        "otx.mpa.modules.models.segmentors.otx_encoder_decoder.OTXEncoderDecoder.simple_test"
    )
    def single_stage_detector__simple_test(ctx, self, img, img_metas, **kwargs):
        # with output activation
        seg_logit = self.inference(img, img_metas, True)
        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(self.feature_map)
            return seg_logit, feature_vector
        else:
            return seg_logit
