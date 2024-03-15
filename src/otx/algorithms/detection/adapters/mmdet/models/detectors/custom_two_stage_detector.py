"""OTX Two Stage Detector Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector

from otx.algorithms.common.utils.task_adapt import map_class_names
from otx.utils.logger import get_logger

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()

# pylint: disable=too-many-locals, unused-argument


@DETECTORS.register_module()
class CustomTwoStageDetector(SAMDetectorMixin, L2SPDetectorMixin, TwoStageDetector):
    """SAM optimizer & L2SP regularizer enabled custom 2-stage detector."""

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

    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=None, **kwargs):
        """Forward function for CustomTwoStageDetector."""
        return super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=gt_bboxes_ignore)

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info(f"----------------- CustomTwoStageDetector.load_state_dict_pre_hook() called w/ prefix: {prefix}")

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
