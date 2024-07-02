# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.roi_heads.roi_extractors.single_level_roi_extractor.py.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
"""

from __future__ import annotations

import torch
from torch import Graph, Tensor
from torch.autograd import Function

from .base_roi_extractor import BaseRoIExtractor

# ruff: noqa: ARG004


class SingleRoIExtractorOpenVINO(Function):
    """This class adds support for ExperimentalDetectronROIFeatureExtractor when exporting to OpenVINO.

    The `forward` method returns the original output, which is calculated in
    advance and added to the SingleRoIExtractorOpenVINO class. In addition, the
    list of arguments is changed here to be more suitable for
    ExperimentalDetectronROIFeatureExtractor.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(
        g: Graph,
        output_size: int,
        featmap_strides: int,
        sample_num: int,
        rois: torch.Value,
        *feats: tuple[torch.Value],
    ) -> Tensor:
        """Run forward."""
        return SingleRoIExtractorOpenVINO.origin_output

    @staticmethod
    def symbolic(
        g: Graph,
        output_size: int,
        featmap_strides: list[int],
        sample_num: int,
        rois: torch.Value,
        *feats: tuple[torch.Value],
    ) -> Graph:
        """Symbolic function for creating onnx op."""
        from torch.onnx.symbolic_opset10 import _slice

        rois = _slice(g, rois, axes=[1], starts=[1], ends=[5])
        domain = "org.openvinotoolkit"
        op_name = "ExperimentalDetectronROIFeatureExtractor"
        return g.op(
            f"{domain}::{op_name}",
            rois,
            *feats,
            output_size_i=output_size,
            pyramid_scales_i=featmap_strides,
            sampling_ratio_i=sample_num,
            image_id_i=0,
            distribute_rois_between_levels_i=1,
            preserve_rois_order_i=0,
            aligned_i=1,
            outputs=1,
        )


class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and
            arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (list[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
            Defaults to 56.
        init_cfg (dict or list[dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        roi_layer: dict,
        out_channels: int,
        featmap_strides: list[int],
        finest_scale: int = 56,
        init_cfg: dict | list[dict] | None = None,
    ) -> None:
        super().__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            init_cfg=init_cfg,
        )
        self.finest_scale = finest_scale

    def map_roi_levels(self, rois: Tensor, num_levels: int) -> Tensor:
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt((rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        return target_lvls.clamp(min=0, max=num_levels - 1).long()

    def forward(self, feats: tuple[Tensor], rois: Tensor, roi_scale_factor: float | None = None) -> Tensor:
        """Extractor ROI feats.

        Args:
            feats (tuple[Tensor]): Multi-scale features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            roi_scale_factor (float, optional): RoI scale factor.
                Defaults to None.

        Returns:
            Tensor: RoI feature.
        """
        # convert fp32 to fp16 when amp is on
        rois = rois.type_as(feats[0])
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        roi_feats = feats[0].new_zeros(rois.size(0), self.out_channels, out_size, out_size)

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(x.view(-1)[0] for x in self.parameters()) * 0.0 + feats[i].sum() * 0.0
        return roi_feats

    def export(
        self,
        feats: tuple[Tensor, ...],
        rois: Tensor,
        roi_scale_factor: float | None = None,
    ) -> Tensor:
        """Export SingleRoIExtractorOpenVINO."""
        # Adding original output to SingleRoIExtractorOpenVINO.
        state = torch._C._get_tracing_state()  # noqa: SLF001
        origin_output = self(feats, rois, roi_scale_factor)
        SingleRoIExtractorOpenVINO.origin_output = origin_output
        torch._C._set_tracing_state(state)  # noqa: SLF001

        output_size = self.roi_layers[0].output_size
        featmap_strides = self.featmap_strides
        sample_num = self.roi_layers[0].sampling_ratio

        args = (output_size, featmap_strides, sample_num, rois, *feats)
        return SingleRoIExtractorOpenVINO.apply(*args)
