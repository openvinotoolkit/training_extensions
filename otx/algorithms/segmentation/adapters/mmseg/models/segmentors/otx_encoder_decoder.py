"""OTX encoder decoder for semantic segmentation."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

import torch
from mmseg.models import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.utils import get_root_logger

from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.utils.task_adapt import map_class_names


# pylint: disable=unused-argument, line-too-long
@SEGMENTORS.register_module()
class OTXEncoderDecoder(EncoderDecoder):
    """OTX encoder decoder."""

    def __init__(self, task_adapt=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        assert task_adapt is not None, "When using task_adapt, task_adapt must be set."

        self._register_load_state_dict_pre_hook(
            functools.partial(
                self.load_state_dict_pre_hook,
                self,  # model
                task_adapt["dst_classes"],  # model_classes
                task_adapt["src_classes"],  # chkpt_classes
            )
        )

    def simple_test(self, img, img_meta, rescale=True, output_logits=False):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        if output_logits:
            seg_pred = seg_logit
        elif self.out_channels == 1:
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

    @staticmethod
    def load_state_dict_pre_hook(
        model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs
    ):  # pylint: disable=too-many-locals, unused-argument
        """Modify input state_dict according to class name matching before weight loading."""
        logger = get_root_logger("INFO")
        logger.info(f"----------------- OTXEncoderDecoder.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        model_dict = model.state_dict()
        param_names = [
            "decode_head.conv_seg.weight",
            "decode_head.conv_seg.bias",
        ]
        for model_name in param_names:
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for model_key, c in enumerate(model2chkpt):
                if c >= 0:
                    model_param[model_key].copy_(chkpt_param[c])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER

    from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (  # pylint: disable=ungrouped-imports
        FeatureVectorHook,
    )

    BASE_CLASS = "otx.algorithms.segmentation.adapters.mmseg.models.segmentors.otx_encoder_decoder.OTXEncoderDecoder"

    @FUNCTION_REWRITER.register_rewriter(f"{BASE_CLASS}.extract_feat")
    def single_stage_detector__extract_feat(ctx, self, img):
        """Extract feature."""
        feat = self.backbone(img)
        self.feature_map = feat
        if self.with_neck:
            feat = self.neck(feat)
        return feat

    @FUNCTION_REWRITER.register_rewriter(f"{BASE_CLASS}.simple_test")
    def single_stage_detector__simple_test(ctx, self, img, img_metas, **kwargs):
        """Test."""
        # with output activation
        seg_logit = self.inference(img, img_metas, True)
        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(self.feature_map)
            return seg_logit, feature_vector
        return seg_logit
