"""OTX CustomVFNet Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.vfnet import VFNet

from otx.algorithms.common.utils.task_adapt import map_class_names
from otx.algorithms.detection.adapters.mmdet.models.detectors.loss_dynamics_mixin import DetLossDynamicsTrackingMixin
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import TrackingLossType
from otx.utils.logger import get_logger

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()

# TODO: Need to fix pylint issues
# pylint: disable=abstract-method, too-many-locals, unused-argument


@DETECTORS.register_module()
class CustomVFNet(SAMDetectorMixin, DetLossDynamicsTrackingMixin, L2SPDetectorMixin, VFNet):
    """SAM optimizer & L2SP regularizer enabled custom VFNet."""

    TRACKING_LOSS_TYPE = (TrackingLossType.cls, TrackingLossType.bbox, TrackingLossType.bbox_refine)

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
        """Forward function for CustomVFNet."""
        return super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=gt_bboxes_ignore)

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info(f"----------------- CustomVFNet.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        model_dict = model.state_dict()
        param_names = [
            "bbox_head.vfnet_cls.weight",
            "bbox_head.vfnet_cls.bias",
        ]
        for model_name in param_names:
            chkpt_name = prefix + model_name
            if model_name not in model_dict or chkpt_name not in chkpt_dict:
                logger.info(f"Skipping weight copy: {chkpt_name}")
                continue

            # Mix weights
            model_param = model_dict[model_name].clone()
            chkpt_param = chkpt_dict[chkpt_name]
            for model_t, ckpt_t in enumerate(model2chkpt):
                if ckpt_t >= 0:
                    model_param[model_t].copy_(chkpt_param[ckpt_t])

            # Replace checkpoint weight by mixed weights
            chkpt_dict[chkpt_name] = model_param
