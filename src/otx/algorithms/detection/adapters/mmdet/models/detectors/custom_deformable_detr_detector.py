"""OTX Deformable DETR Class for mmdetection detectors."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.deformable_detr import DeformableDETR

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    ActivationMapHook,
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.common.utils.task_adapt import map_class_names

logger = get_logger()


@DETECTORS.register_module()
class CustomDeformableDETR(DeformableDETR):
    """Custom Deformable DETR with task adapt.

    Deformable DETR does not support task adapt, so it just take task_adpat argument.
    """

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # ckpt_classes
                )
            )
        self.cls_layers = ["cls_branches"]

    def load_state_dict_pre_hook(self, model_classes, ckpt_classes, ckpt_dict, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info("----------------- CustomDeformableDETR.load_state_dict_pre_hook() called")

        model_classes = list(model_classes)
        ckpt_classes = list(ckpt_classes)
        model2ckpt = map_class_names(model_classes, ckpt_classes)
        logger.info(f"{ckpt_classes} -> {model_classes} ({model2ckpt})")

        model_dict = self.state_dict()
        param_names = []
        for model_name in model_dict:
            for cls_layer in self.cls_layers:
                if cls_layer in model_name:
                    param_names.append(model_name)
        for param_name in param_names:
            ckpt_name = param_name
            if param_name not in model_dict or ckpt_name not in ckpt_dict:
                logger.info(f"Skipping weight copy: {ckpt_name}")
                continue

            # Mix weights
            model_param = model_dict[param_name].clone()
            ckpt_param = ckpt_dict[ckpt_name]
            for model_t, ckpt_t in enumerate(model2ckpt):
                if ckpt_t >= 0:
                    model_param[model_t].copy_(ckpt_param[ckpt_t])

            # Replace checkpoint weight by mixed weights
            ckpt_dict[ckpt_name] = model_param


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
            saliency_map = ActivationMapHook(self).func(cls_scores)
            return (*bbox_results, feature_vector, saliency_map)

        return bbox_results
