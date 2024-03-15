"""Detection Saliency Map Hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmdet.core import bbox2roi

from otx.algorithms.common.adapters.mmcv.hooks.recording_forward_hook import (
    BaseRecordingForwardHook,
)
from otx.algorithms.detection.adapters.mmdet.models.heads.custom_atss_head import (
    CustomATSSHead,
)
from otx.algorithms.detection.adapters.mmdet.models.heads.custom_ssd_head import (
    CustomSSDHead,
)
from otx.algorithms.detection.adapters.mmdet.models.heads.custom_vfnet_head import (
    CustomVFNetHead,
)
from otx.algorithms.detection.adapters.mmdet.models.heads.custom_yolox_head import (
    CustomYOLOXHead,
)

# pylint: disable=too-many-locals


class DetClassProbabilityMapHook(BaseRecordingForwardHook):
    """Saliency map hook for object detection models."""

    def __init__(self, module: torch.nn.Module, normalize: bool = True, use_cls_softmax: bool = True) -> None:
        super().__init__(module, normalize=normalize)
        self._neck = module.neck if module.with_neck else None
        self._bbox_head = module.bbox_head
        self._num_cls_out_channels = module.bbox_head.cls_out_channels  # SSD-like heads also have background class
        if hasattr(module.bbox_head, "anchor_generator"):
            self._num_anchors = module.bbox_head.anchor_generator.num_base_anchors
        else:
            self._num_anchors = [1] * 10
        self.use_cls_softmax = use_cls_softmax

    def func(
        self,
        feature_map: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        _: int = -1,
        cls_scores_provided: bool = False,
    ) -> torch.Tensor:
        """Generate the saliency map from raw classification head output, then normalizing to (0, 255).

        :param x: Feature maps from backbone/FPN or classification scores from cls_head
        :param cls_scores_provided: If True - use 'x' as is, otherwise forward 'x' through the classification head
        :return: Class-wise Saliency Maps. One saliency map per each class - [batch, class_id, H, W]
        """
        if cls_scores_provided:
            cls_scores = feature_map
        else:
            cls_scores = self._get_cls_scores_from_feature_map(feature_map)

        middle_idx = len(cls_scores) // 2
        # resize to the middle feature map
        batch_size, _, height, width = cls_scores[middle_idx].size()
        saliency_maps = torch.empty(batch_size, self._num_cls_out_channels, height, width)
        for batch_idx in range(batch_size):
            cls_scores_anchorless = []
            for scale_idx, cls_scores_per_scale in enumerate(cls_scores):
                cls_scores_anchor_grouped = cls_scores_per_scale[batch_idx].reshape(
                    self._num_anchors[scale_idx], (self._num_cls_out_channels), *cls_scores_per_scale.shape[-2:]
                )
                cls_scores_out, _ = cls_scores_anchor_grouped.max(dim=0)
                cls_scores_anchorless.append(cls_scores_out.unsqueeze(0))
            cls_scores_anchorless_resized = []
            for cls_scores_anchorless_per_level in cls_scores_anchorless:
                cls_scores_anchorless_resized.append(
                    F.interpolate(cls_scores_anchorless_per_level, (height, width), mode="bilinear")
                )
            saliency_maps[batch_idx] = torch.cat(cls_scores_anchorless_resized, dim=0).mean(dim=0)

        # Don't use softmax for tiles in tiling detection, if the tile doesn't contain objects,
        # it would highlight one of the class maps as a background class
        if self.use_cls_softmax:
            saliency_maps[0] = torch.stack([torch.softmax(t, dim=1) for t in saliency_maps[0]])

        if self._norm_saliency_maps:
            saliency_maps = saliency_maps.reshape((batch_size, self._num_cls_out_channels, -1))
            saliency_maps = self._normalize_map(saliency_maps)

        saliency_maps = saliency_maps.reshape((batch_size, self._num_cls_out_channels, height, width))

        return saliency_maps

    def _get_cls_scores_from_feature_map(self, x: torch.Tensor) -> List:
        """Forward features through the classification head of the detector."""
        with torch.no_grad():
            if self._neck is not None:
                x = self._neck(x)

            if isinstance(self._bbox_head, CustomSSDHead):
                cls_scores = []
                for feat, cls_conv in zip(x, self._bbox_head.cls_convs):
                    cls_scores.append(cls_conv(feat))
            elif isinstance(self._bbox_head, CustomATSSHead):
                cls_scores = []
                for cls_feat in x:
                    for cls_conv in self._bbox_head.cls_convs:
                        cls_feat = cls_conv(cls_feat)
                    cls_score = self._bbox_head.atss_cls(cls_feat)
                    cls_scores.append(cls_score)
            elif isinstance(self._bbox_head, CustomVFNetHead):
                # Not clear how to separate cls_scores from bbox_preds
                cls_scores, _, _ = self._bbox_head(x)
            elif isinstance(self._bbox_head, CustomYOLOXHead):

                def forward_single(x, cls_convs, conv_cls):
                    """Forward feature of a single scale level."""
                    cls_feat = cls_convs(x)
                    cls_score = conv_cls(cls_feat)
                    return cls_score

                map_results = map(
                    forward_single, x, self._bbox_head.multi_level_cls_convs, self._bbox_head.multi_level_conv_cls
                )
                cls_scores = list(map_results)
            else:
                raise NotImplementedError(
                    "Not supported detection head provided. "
                    "DetClassProbabilityMap supports only the following single stage detectors: "
                    "YOLOXHead, ATSSHead, SSDHead, VFNetHead."
                )
        return cls_scores


class MaskRCNNRecordingForwardHook(BaseRecordingForwardHook):
    """Saliency map hook for Mask R-CNN model. Only for torch model, does not support OpenVINO IR model.

    Args:
        module (torch.nn.Module): Mask R-CNN model.
        input_img_shape (Tuple[int]): Resolution of the model input image.
        saliency_map_shape (Tuple[int]): Resolution of the output saliency map.
        max_detections_per_img (int): Upper limit of the number of detections
            from which soft mask predictions are getting aggregated.
        normalize (bool): Flag that defines if the output saliency map will be normalized.
            Although, partial normalization is anyway done by segmentation mask head.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        input_img_shape: Tuple[int, int],
        saliency_map_shape: Tuple[int, int] = (224, 224),
        max_detections_per_img: int = 300,
        normalize: bool = True,
    ) -> None:
        super().__init__(module)
        self._neck = module.neck if module.with_neck else None
        self._input_img_shape = input_img_shape
        self._saliency_map_shape = saliency_map_shape
        self._max_detections_per_img = max_detections_per_img
        self._norm_saliency_maps = normalize

    def func(
        self,
        feature_map: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        _: int = -1,
    ) -> List[List[Optional[np.ndarray]]]:
        """Generate saliency maps by aggregating per-class soft predictions of mask head for all detected boxes.

        :param feature_map: Feature maps from backbone.
        :return: Class-wise Saliency Maps. One saliency map per each predicted class.
        """
        with torch.no_grad():
            if self._neck is not None:
                feature_map = self._module.neck(feature_map)

            det_bboxes, det_labels = self._get_detections(feature_map)
            saliency_maps = self._get_saliency_maps_from_mask_predictions(feature_map, det_bboxes, det_labels)
            if self._norm_saliency_maps:
                saliency_maps = self._normalize(saliency_maps)
        return saliency_maps

    def _get_detections(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        batch_size = x[0].shape[0]
        img_metas = [
            {
                "scale_factor": [1, 1, 1, 1],  # dummy scale_factor, not used
                "img_shape": self._input_img_shape,
            }
        ]
        img_metas *= batch_size
        proposals = self._module.rpn_head.simple_test_rpn(x, img_metas)
        test_cfg = copy.deepcopy(self._module.roi_head.test_cfg)
        test_cfg["max_per_img"] = self._max_detections_per_img
        test_cfg["nms"]["iou_threshold"] = 1
        test_cfg["nms"]["max_num"] = self._max_detections_per_img
        det_bboxes, det_labels = self._module.roi_head.simple_test_bboxes(
            x, img_metas, proposals, test_cfg, rescale=False
        )
        return det_bboxes, det_labels

    def _get_saliency_maps_from_mask_predictions(
        self, x: torch.Tensor, det_bboxes: List[torch.Tensor], det_labels: List[torch.Tensor]
    ) -> List[List[Optional[np.ndarray]]]:
        _bboxes = [det_bboxes[i][:, :4] for i in range(len(det_bboxes))]
        mask_rois = bbox2roi(_bboxes)
        mask_results = self._module.roi_head._mask_forward(x, mask_rois)
        mask_pred = mask_results["mask_pred"]
        num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
        mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

        batch_size = x[0].shape[0]

        scale_x = self._input_img_shape[1] / self._saliency_map_shape[1]
        scale_y = self._input_img_shape[0] / self._saliency_map_shape[0]
        scale_factor = torch.FloatTensor((scale_x, scale_y, scale_x, scale_y))
        test_cfg = self._module.roi_head.test_cfg.copy()
        test_cfg["mask_thr_binary"] = -1

        saliency_maps = [[None for _ in range(self._module.roi_head.mask_head.num_classes)] for _ in range(batch_size)]

        for i in range(batch_size):
            if det_bboxes[i].shape[0] == 0:
                continue
            else:
                segm_result = self._module.roi_head.mask_head.get_seg_masks(
                    mask_preds[i],
                    _bboxes[i],
                    det_labels[i],
                    test_cfg,
                    self._saliency_map_shape,
                    scale_factor=scale_factor,
                    rescale=True,
                )
                for class_id, segm_res in enumerate(segm_result):
                    if segm_res:
                        saliency_maps[i][class_id] = np.mean(np.array(segm_res), axis=0)
        return saliency_maps

    @staticmethod
    def _normalize(saliency_maps: List[List[Optional[np.ndarray]]]) -> List[List[Optional[np.ndarray]]]:
        batch_size = len(saliency_maps)
        num_classes = len(saliency_maps[0])
        for i in range(batch_size):
            for class_id in range(num_classes):
                per_class_map = saliency_maps[i][class_id]
                if per_class_map is not None:
                    max_values = np.max(per_class_map)
                    per_class_map = 255 * (per_class_map) / (max_values + 1e-12)
                    per_class_map = per_class_map.astype(np.uint8)
                    saliency_maps[i][class_id] = per_class_map
        return saliency_maps
