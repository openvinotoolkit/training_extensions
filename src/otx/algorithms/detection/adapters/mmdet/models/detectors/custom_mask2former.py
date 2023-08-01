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
    import torch.nn.functional as F
    from mmdeploy.codebase.mmdet.deploy import get_post_processing_params
    from mmdeploy.core import FUNCTION_REWRITER, mark
    from mmdeploy.utils import is_dynamic_shape

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
        out = self.panoptic_fusion_head.simple_test(mask_cls_results, mask_pred_results, img_metas, **kwargs)

        if ctx.cfg["dump_features"]:
            feature_vector = FeatureVectorHook.func(feats)
            # Saliency map will be generated from predictions. Generate dummy saliency_map.
            saliency_map = torch.empty(1, dtype=torch.uint8)
            return (*out, feature_vector, saliency_map)

        return out

    @mark("custom_mask2former_forward", inputs=["input"], outputs=["dets", "labels", "masks", "feats", "saliencies"])
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
        "otx.algorithms.detection.adapters.mmdet.models.detectors.custom_mask2former.CustomMask2Former.forward"
    )
    def custom_mask2former__forward(ctx, self, img, img_metas=None, return_loss=False, **kwargs):
        """Internal Function for __forward for CustomMaskRCNN."""
        if img_metas is None:
            img_metas = [{}]
        else:
            assert len(img_metas) == 1, "do not support aug_test"
            img_metas = img_metas[0]

        if isinstance(img, list):
            img = img[0]

        return __forward_impl(ctx, self, img, img_metas=img_metas, **kwargs)

    @FUNCTION_REWRITER.register_rewriter(
        "otx.algorithms.detection.adapters.mmdet.models.heads.custom_maskformer_fusion_head.CustomMaskFormerFusionHead.simple_test"
    )
    def custom_maskformer_fusion_head__simple_test(
        ctx, self, mask_cls_results, mask_pred_results, img_metas, rescale=False, **kwargs
    ):
        batch_size = mask_cls_results.size(0)
        assert batch_size == 1, "do not support batch size > 1"
        mask_cls_result = mask_cls_results[0]
        mask_pred_result = mask_pred_results[0]
        img_meta = img_metas[0]

        # remove padding
        img_height, img_width = img_meta["img_shape"][:2]
        mask_pred_result = mask_pred_result[:, :img_height, :img_width]

        post_params = get_post_processing_params(ctx.cfg)
        export_postprocess_mask = post_params.get("export_postprocess_mask", True)
        if export_postprocess_mask:
            # return result in original resolution
            ori_height, ori_width = img_meta["ori_shape"][:2]
            mask_pred_result = F.interpolate(
                mask_pred_result[:, None], size=(ori_height, ori_width), mode="bilinear", align_corners=False
            )[:, 0]

        """Custom instance_postprocess for MaskFormerFusionHead."""
        max_per_image = self.test_cfg.get("max_per_image", 100)
        num_queries = mask_cls_result.size(0)
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls_result, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(max_per_image, sorted=False)
        labels = labels[top_indices]

        query_indices = top_indices // self.num_classes
        mask_pred_result = mask_pred_result[query_indices]

        masks = (mask_pred_result > 0).float()
        mask_scores_per_image = (mask_pred_result.sigmoid() * masks).flatten(1).sum(1) / (
            masks.flatten(1).sum(1) + 1e-6
        )
        det_scores = scores_per_image * mask_scores_per_image
        masks = masks.bool()

        # filter by score
        keep = det_scores > self.test_cfg.score_threshold
        det_scores = det_scores[keep]
        masks = masks[keep]
        labels = labels[keep]

        bboxes = det_scores.new_zeros(batch_size, det_scores.shape[-1], 2)
        # full screen box so we can postprocess mask outside the model
        bboxes = torch.cat(
            [
                bboxes,
                bboxes.new_full((*bboxes.shape[:2], 1), img_width),
                bboxes.new_full((*bboxes.shape[:2], 1), img_height),
            ],
            dim=-1,
        )
        labels = labels.reshape(batch_size, -1)
        dets = torch.cat([bboxes, det_scores.reshape(batch_size, -1, 1)], dim=-1)
        masks = masks.unsqueeze(0)
        return dets, labels, masks
