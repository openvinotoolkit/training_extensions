# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.roi_heads.roi_extractors.base_roi_extractor.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/roi_heads/roi_extractors/base_roi_extractor.py
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor, nn
from torchvision.ops import RoIAlign

from otx.algo.modules.base_module import BaseModule


class BaseRoIExtractor(BaseModule, metaclass=ABCMeta):
    """Base class for RoI extractor.

    Args:
        roi_layer (nn.Module): RoI layer module.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (list[int]): Strides of input feature maps.
        init_cfg (dict or list[dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        roi_layer: nn.Module,
        out_channels: int,
        featmap_strides: list[int],
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.roi_layers = self.build_roi_layers(roi_layer, featmap_strides)
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides

    @property
    def num_inputs(self) -> int:
        """int: Number of input feature maps."""
        return len(self.featmap_strides)

    def build_roi_layers(self, roi_layer: nn.Module, featmap_strides: list[int]) -> nn.ModuleList:
        """Build RoI operator to extract feature from each level feature map.

        Args:
            layer_cfg (nn.Module): RoI layer module such as ``RoIAlign``.
            featmap_strides (list[int]): The stride of input feature map w.r.t
                to the original image size, which would be used to scale RoI
                coordinate (original image coordinate system) to feature
                coordinate system.

        Returns:
            nn.ModuleList: The RoI extractor modules for each level feature map.
        """
        if not isinstance(roi_layer, RoIAlign):
            msg = f"Unsupported RoI layer type {roi_layer.__name__}"
            raise TypeError(msg)
        return nn.ModuleList(
            [
                RoIAlign(
                    spatial_scale=1 / s,
                    output_size=roi_layer.output_size,
                    sampling_ratio=roi_layer.sampling_ratio,
                    aligned=roi_layer.aligned,
                )
                for s in featmap_strides
            ],
        )

    def roi_rescale(self, rois: Tensor, scale_factor: float) -> Tensor:
        """Scale RoI coordinates by scale factor.

        Args:
            rois (Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            Tensor: Scaled RoI.
        """
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        return torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)

    @abstractmethod
    def forward(self, feats: tuple[Tensor], rois: Tensor, roi_scale_factor: float | None = None) -> Tensor:
        """Extractor ROI feats.

        Args:
            feats (Tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            roi_scale_factor (float, optional): RoI scale factor.
                Defaults to None.

        Returns:
            Tensor: RoI feature.
        """
