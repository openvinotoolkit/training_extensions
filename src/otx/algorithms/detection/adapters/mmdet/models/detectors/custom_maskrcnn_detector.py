"""OTX MaskRCNN Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.mask_rcnn import MaskRCNN

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.common.utils.task_adapt import map_class_names

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()

# pylint: disable=too-many-locals, protected-access, unused-argument


@DETECTORS.register_module()
class CustomMaskRCNN(SAMDetectorMixin, L2SPDetectorMixin, MaskRCNN):
    """CustomMaskRCNN Class for mmdetection detectors."""

    def __init__(self, *args, task_adapt=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Hook for class-sensitive weight loading
        if task_adapt:
            self._register_load_state_dict_pre_hook(
                functools.partial(
                    self.load_state_dict_pre_hook,
                    self,  # model
                    task_adapt["dst_classes"],  # model_classes
                    task_adapt["src_classes"],  # chkpt_classes
                )
            )

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info(f"----------------- CustomMaskRCNN.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_dict = model.state_dict()
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        # List of class-relevant params & their row-stride
        param_strides = {
            "roi_head.bbox_head.fc_cls.weight": 1,
            "roi_head.bbox_head.fc_cls.bias": 1,
            "roi_head.bbox_head.fc_reg.weight": 4,  # 4 rows (bbox) for each class
            "roi_head.bbox_head.fc_reg.bias": 4,
        }

        for model_name, stride in param_strides.items():
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for model_t, ckpt_t in enumerate(model2chkpt):
                if ckpt_t >= 0:
                    # Copying only matched weight rows
                    model_param[(model_t) * stride : (model_t + 1) * stride].copy_(
                        chkpt_param[(ckpt_t) * stride : (ckpt_t + 1) * stride]
                    )
            if model_param.shape[0] > len(model_classes * stride):  # BG class
                num_ckpt_class = len(chkpt_classes)
                num_model_class = len(model_classes)
                model_param[(num_model_class) * stride : (num_model_class + 1) * stride].copy_(
                    chkpt_param[(num_ckpt_class) * stride : (num_ckpt_class + 1) * stride]
                )

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER, mark
    from mmdeploy.utils import is_dynamic_shape

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_maskrcnn_detector.CustomMaskRCNN.simple_test"
    )
    def custom_mask_rcnn__simple_test(ctx, self, img, img_metas, proposals=None, **kwargs):
        """Function for custom_mask_rcnn__simple_test."""
        assert self.with_bbox, "Bbox head must be implemented."
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        if proposals is None:
            proposals, _ = self.rpn_head.simple_test_rpn(x, img_metas)
        out = self.roi_head.simple_test(x, proposals, img_metas, rescale=False)

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(x)
            # Saliency map will be generated from predictions. Generate dummy saliency_map.
            saliency_map = torch.empty(1, dtype=torch.uint8)
            return (*out, feature_vector, saliency_map)

        return out

    @mark("custom_maskrcnn_forward", inputs=["input"], outputs=["dets", "labels", "masks", "feats", "saliencies"])
    def __forward_impl(ctx, self, img, img_metas, **kwargs):
        """Internal Function for __forward_impl."""
        assert isinstance(img, torch.Tensor)

        deploy_cfg = ctx.cfg
        is_dynamic_flag = is_dynamic_shape(deploy_cfg)
        # get origin input shape as tensor to support onnx dynamic shape
        img_shape = torch._shape_as_tensor(img)[2:]
        if not is_dynamic_flag:
            img_shape = [int(val) for val in img_shape]
        img_metas[0]["img_shape"] = img_shape
        return self.simple_test(img, img_metas, **kwargs)

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_maskrcnn_detector.CustomMaskRCNN.forward"
    )
    def custom_maskrcnn__forward(ctx, self, img, img_metas=None, return_loss=False, **kwargs):
        """Internal Function for __forward for CustomMaskRCNN."""
        if img_metas is None:
            img_metas = [{}]
        else:
            assert len(img_metas) == 1, "do not support aug_test"
            img_metas = img_metas[0]

        if isinstance(img, list):
            img = img[0]

        return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)
