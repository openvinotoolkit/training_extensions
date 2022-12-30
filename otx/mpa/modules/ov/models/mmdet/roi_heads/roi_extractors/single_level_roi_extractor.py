# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdet.models.builder import ROI_EXTRACTORS
from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor import (
    SingleRoIExtractor as OriginSingleRoIExtractor,
)
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair


@ROI_EXTRACTORS.register_module(force=True)
class SingleRoIExtractor(OriginSingleRoIExtractor):
    def build_roi_layers(self, layer_cfg, featmap_strides):

        if layer_cfg["type"] == "RoIInterpolationPool":
            cfg = layer_cfg.copy()
            cfg.pop("type")
            return nn.ModuleList([RoIInterpolationPool(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        else:
            return super().build_roi_layers(layer_cfg, featmap_strides)


class RoIInterpolationPool(nn.Module):
    def __init__(self, output_size, spatial_scale, mode="bilinear"):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.mode = mode

    def forward(self, input, rois):
        outs = []
        for roi in rois:
            batch_idx = roi[0].to(dtype=torch.int)
            roi = roi[1:]
            roi = roi * self.spatial_scale
            x1, y1 = roi[:2].floor().to(dtype=torch.int)
            x2, y2 = roi[2:].ceil().to(dtype=torch.int)
            outs.append(
                F.interpolate(
                    input[batch_idx : batch_idx + 1, :, y1:y2, x1:x2],
                    self.output_size,
                    mode=self.mode,
                    align_corners=True,
                )
            )
        outs = torch.cat(outs, 0)
        return outs

    def __repr__(self):
        s = self.__class__.__name__
        s += f"(output_size={self.output_size}, "
        s += f"spatial_scale={self.spatial_scale}, "
        s += f"mode={self.mode})"
        return s
