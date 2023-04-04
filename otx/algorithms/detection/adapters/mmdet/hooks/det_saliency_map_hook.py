"""Detection Saliency Map Hook."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

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


class DetSaliencyMapHook(BaseRecordingForwardHook):
    """Saliency map hook for object detection models."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__(module)
        self._neck = module.neck if module.with_neck else None
        self._bbox_head = module.bbox_head
        self._num_cls_out_channels = module.bbox_head.cls_out_channels  # SSD-like heads also have background class
        if hasattr(module.bbox_head, "anchor_generator"):
            self._num_anchors = module.bbox_head.anchor_generator.num_base_anchors
        else:
            self._num_anchors = [1] * 10

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

        batch_size, _, height, width = cls_scores[-1].size()
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

        saliency_maps = saliency_maps.reshape((batch_size, self._num_cls_out_channels, -1))
        max_values, _ = torch.max(saliency_maps, -1)
        min_values, _ = torch.min(saliency_maps, -1)
        saliency_maps = 255 * (saliency_maps - min_values[:, :, None]) / (max_values - min_values + 1e-12)[:, :, None]
        saliency_maps = saliency_maps.reshape((batch_size, self._num_cls_out_channels, height, width))
        saliency_maps = saliency_maps.to(torch.uint8)
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
                    "DetSaliencyMapHook supports only the following single stage detectors: "
                    "YOLOXHead, ATSSHead, SSDHead, VFNetHead."
                )
        return cls_scores
