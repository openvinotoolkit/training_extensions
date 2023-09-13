"""OTX RTMDet Class for mmdetection detectors."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import functools
from typing import List

import numpy as np
import torch
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector
from mmengine.structures import InstanceData

from otx.algorithms.common.adapters.mmdeploy.utils import is_mmdeploy_enabled
from otx.algorithms.common.utils.logger import get_logger
from otx.algorithms.common.utils.task_adapt import map_class_names

logger = get_logger()


def pack_gt_instances(gt_masks, gt_labels, gt_bboxes) -> List[InstanceData]:
    """Pack ground truth instances into a list of InstanceData.

    Args:
        gt_masks (Tensor): ground truth masks.
        gt_labels (Tensor): ground truth labels.
        gt_bboxes (Tensor): ground truth bounding boxes.

    Returns:
        list[InstanceData]: list of InstanceData.
    """
    batch_gt_instances = []
    for gt_mask, gt_label, gt_bbox in zip(gt_masks, gt_labels, gt_bboxes):
        gt_instance = InstanceData()
        gt_instance.masks = gt_mask
        gt_instance.labels = gt_label
        gt_instance.bboxes = gt_bbox
        batch_gt_instances.append(gt_instance)
    return batch_gt_instances


@DETECTORS.register_module()
class CustomRTMDetInst(SingleStageDetector):
    """Implementation of RTMDet."""

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
        logger.info(f"----------------- CustomRTMDet.load_state_dict_pre_hook() called w/ prefix: {prefix}")

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

    def forward_train(self, img, img_metas, gt_masks, gt_labels, gt_bboxes, gt_bboxes_ignore=None, **kwargs):
        """Forward function for training."""
        gt_masks = [gt_mask.to_tensor(dtype=torch.bool, device=img.device) for gt_mask in gt_masks]
        x = self.extract_feat(img)
        bbox_head_preds = self.bbox_head(x)

        batch_gt_instances = pack_gt_instances(gt_masks, gt_labels, gt_bboxes)

        losses = self.bbox_head.loss(
            *bbox_head_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=img_metas,
            batch_gt_instances_ignore=gt_bboxes_ignore,
        )
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)

        with_mask = True if "masks" in results_list[0] else False

        if with_mask:
            return [self.format_mask_results(results) for results in results_list]
        return self.format_bbox_results(results_list)

    def format_bbox_results(self, results_list):
        """Format box results."""
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes) for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def format_mask_results(self, results):
        """Format the model predictions according to the interface with dataset.

        Args:
            results (:obj:`InstanceData`): Processed
                results of single images. Usually contains
                following keys.

                - scores (Tensor): Classification scores, has shape
                  (num_instance,)
                - labels (Tensor): Has shape (num_instances,).
                - masks (Tensor): Processed mask results, has
                  shape (num_instances, h, w).

        Returns:
            tuple: Formatted bbox and mask results.. It contains two items:

                - bbox_results (list[np.ndarray]): BBox results of
                  single image. The list corresponds to each class.
                  each ndarray has a shape (N, 5), N is the number of
                  bboxes with this category, and last dimension
                  5 arrange as (x1, y1, x2, y2, scores).
                - mask_results (list[np.ndarray]): Mask results of
                  single image. The list corresponds to each class.
                  each ndarray has shape (N, img_h, img_w), N
                  is the number of masks with this category.
        """
        data_keys = results.keys()
        assert "scores" in data_keys
        assert "labels" in data_keys

        assert "masks" in data_keys, "results should contain " "masks when format the results "
        mask_results = [[] for _ in range(self.bbox_head.num_classes)]

        num_masks = len(results)

        if num_masks == 0:
            bbox_results = [np.zeros((0, 5), dtype=np.float32) for _ in range(self.bbox_head.num_classes)]
            return bbox_results, mask_results

        labels = results.labels.detach().cpu().numpy()

        det_bboxes = torch.cat([results.bboxes, results.scores[:, None]], dim=-1)
        det_bboxes = det_bboxes.detach().cpu().numpy()
        bbox_results = [det_bboxes[labels == i, :] for i in range(self.bbox_head.num_classes)]

        masks = results.masks.detach().cpu().numpy()

        for idx in range(num_masks):
            mask = masks[idx]
            mask_results[labels[idx]].append(mask)

        return bbox_results, mask_results


if is_mmdeploy_enabled():
    from mmdeploy.core import FUNCTION_REWRITER, mark
    from mmdeploy.utils import is_dynamic_shape

    from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import FeatureVectorHook
    from otx.algorithms.detection.adapters.mmdet.hooks.det_class_probability_map_hook import (
        DetClassProbabilityMapHook,
    )

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_rtmdet.CustomRTMDetInst.simple_test"
    )
    def custom_rtmdet_inst__simple_test(ctx, self, img, img_metas, rescale=False):
        """Rewritten CustomRTMDetInst.simple_test."""
        feat = self.extract_feat(img)
        outs = self.bbox_head(feat)
        mask_results = self.bbox_head.get_bboxes(*outs, cfg=self.test_cfg, img_metas=img_metas, rescale=rescale)

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
            return (*mask_results, feature_vector, saliency_map)

        return mask_results

    @mark("custom_rtmdet_inst_forward", inputs=["input"], outputs=["dets", "labels", "masks", "feats", "saliencies"])
    def __forward_impl(ctx, self, img, img_metas, **kwargs):
        """Rewritten CustomRTMDetInst.forward."""
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
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_rtmdet.CustomRTMDetInst.forward"
    )
    def custom_rtmdet_inst__forward(ctx, self, img, img_metas=None, return_loss=False, **kwargs):
        """Rewritten CustomRTMDetInst.forward."""
        if img_metas is None:
            img_metas = [{}]
        else:
            assert len(img_metas) == 1, "do not support aug_test"
            img_metas = img_metas[0]

        if isinstance(img, list):
            img = img[0]

        return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)
