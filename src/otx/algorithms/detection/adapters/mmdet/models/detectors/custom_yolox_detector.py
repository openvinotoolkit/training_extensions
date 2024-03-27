"""OTX YOLOX Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import functools

import torch
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.yolox import YOLOX

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    FeatureVectorHook,
)
from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.utils.task_adapt import map_class_names
from otx.algorithms.detection.adapters.mmdet.hooks.det_class_probability_map_hook import (
    DetClassProbabilityMapHook,
)
from otx.algorithms.detection.adapters.mmdet.models.detectors.loss_dynamics_mixin import (
    DetLossDynamicsTrackingMixin,
)
from otx.algorithms.detection.adapters.mmdet.models.loss_dyns import TrackingLossType
from otx.utils.logger import get_logger

from .l2sp_detector_mixin import L2SPDetectorMixin
from .sam_detector_mixin import SAMDetectorMixin

logger = get_logger()

# TODO: Need to fix pylint issues
# pylint: disable=too-many-locals, unused-argument, protected-access, abstract-method


@DETECTORS.register_module()
class CustomYOLOX(SAMDetectorMixin, DetLossDynamicsTrackingMixin, L2SPDetectorMixin, YOLOX):
    """SAM optimizer & L2SP regularizer enabled custom YOLOX."""

    TRACKING_LOSS_TYPE = (TrackingLossType.cls, TrackingLossType.bbox)

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
        """Forward function for CustomYOLOX."""
        return super().forward_train(img, img_metas, gt_bboxes, gt_labels, gt_bboxes_ignore=gt_bboxes_ignore)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    @staticmethod
    def load_state_dict_pre_hook(model, model_classes, chkpt_classes, chkpt_dict, prefix, *args, **kwargs):
        """Modify input state_dict according to class name matching before weight loading."""
        logger.info(f"----------------- CustomYOLOX.load_state_dict_pre_hook() called w/ prefix: {prefix}")

        # Dst to src mapping index
        model_classes = list(model_classes)
        chkpt_classes = list(chkpt_classes)
        model2chkpt = map_class_names(model_classes, chkpt_classes)
        logger.info(f"{chkpt_classes} -> {model_classes} ({model2chkpt})")

        model_dict = model.state_dict()
        param_names = [
            "bbox_head.multi_level_conv_cls.0.weight",
            "bbox_head.multi_level_conv_cls.0.bias",
            "bbox_head.multi_level_conv_cls.1.weight",
            "bbox_head.multi_level_conv_cls.1.bias",
            "bbox_head.multi_level_conv_cls.2.weight",
            "bbox_head.multi_level_conv_cls.2.bias",
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

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]["img_shape_for_onnx"] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]["pad_shape_for_onnx"] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)

        # FIXME: mmdet does not support yolox onnx export for now
        # This is a temporary workaround
        # https://github.com/open-mmlab/mmdetection/issues/6487
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)[0]

        return det_bboxes, det_labels


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER, mark
    from mmdeploy.utils import is_dynamic_shape

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_yolox_detector.CustomYOLOX.simple_test"
    )
    def custom_yolox__simple_test(ctx, self, img, img_metas, **kwargs):
        """Function for custom_yolox__simple_test."""
        feat = self.extract_feat(img)
        outs = self.bbox_head(feat)
        bbox_results = self.bbox_head.get_bboxes(*outs, img_metas=img_metas, cfg=self.test_cfg, **kwargs)

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(feat)
            cls_scores = outs[0]
            postprocess_kwargs = {
                "use_cls_softmax": ctx.cfg["softmax_saliency_maps"],
                "normalize": ctx.cfg["normalize_saliency_maps"],
            }
            saliency_map = DetClassProbabilityMapHook(self, **postprocess_kwargs).func(
                cls_scores, cls_scores_provided=True
            )
            return (*bbox_results, feature_vector, saliency_map)

        return bbox_results

    @mark("custom_yolox_forward", inputs=["input"], outputs=["dets", "labels", "feats", "saliencies"])
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
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_yolox_detector.CustomYOLOX.forward"
    )
    def custom_yolox__forward(ctx, self, img, img_metas=None, return_loss=False, **kwargs):
        """Internal Function for __forward for CustomYOLOX."""
        if img_metas is None:
            img_metas = [{}]
        else:
            assert len(img_metas) == 1, "do not support aug_test"
            img_metas = img_metas[0]

        if isinstance(img, list):
            img = img[0]

        return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)
