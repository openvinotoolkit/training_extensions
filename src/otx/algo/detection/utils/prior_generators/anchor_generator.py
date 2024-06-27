# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdet.models.task_modules.prior_generators.anchor_generator.

Reference : https://github.com/open-mmlab/mmdetection/blob/v3.2.0/mmdet/models/task_modules/prior_generators/anchor_generator.py
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair

from otx.algo.common.utils.prior_generators import AnchorGenerator


class SSDAnchorGeneratorClustered(AnchorGenerator):
    """Custom Anchor Generator for SSD."""

    def __init__(
        self,
        strides: list[int],
        widths: list[list[float]],
        heights: list[list[float]],
    ) -> None:
        """Initialize SSDAnchorGeneratorClustered.

        Args:
            strides (tuple[int]): Anchor's strides.
            widths (list[list[int]]): Anchor's widths.
            heights (list[list[int]]): Anchor's height.
        """
        self.strides = [_pair(stride) for stride in strides]
        self.widths = widths
        self.heights = heights
        self.centers: list[tuple[float, float]] = [(stride / 2.0, stride / 2.0) for stride in strides]

        self.center_offset = 0
        self.gen_base_anchors()

    def gen_base_anchors(self) -> None:  # type: ignore[override]
        """Generate base anchor for SSD."""
        multi_level_base_anchors = []
        for widths, heights, centers in zip(self.widths, self.heights, self.centers):
            base_anchors = self.gen_single_level_base_anchors(
                widths=Tensor(widths),
                heights=Tensor(heights),
                center=Tensor(centers),
            )
            multi_level_base_anchors.append(base_anchors)
        self.base_anchors = multi_level_base_anchors

    def gen_single_level_base_anchors(  # type: ignore[override]
        self,
        widths: Tensor,
        heights: Tensor,
        center: Tensor,
    ) -> Tensor:
        """Generate single_level_base_anchors for SSD.

        Args:
            widths (Tensor): Widths of base anchors.
            heights (Tensor): Heights of base anchors.
            center (Tensor): Centers of base anchors.
        """
        x_center, y_center = center

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * widths,
            y_center - 0.5 * heights,
            x_center + 0.5 * widths,
            y_center + 0.5 * heights,
        ]
        return torch.stack(base_anchors, dim=-1)

    def __repr__(self) -> str:
        """Str: a string that describes the module."""
        indent_str = "    "
        repr_str = self.__class__.__name__ + "(\n"
        repr_str += f"{indent_str}strides={self.strides},\n"
        repr_str += f"{indent_str}widths={self.widths},\n"
        repr_str += f"{indent_str}heights={self.heights},\n"
        repr_str += f"{indent_str}num_levels={self.num_levels}\n"
        repr_str += f"{indent_str}centers={self.centers},\n"
        repr_str += f"{indent_str}center_offset={self.center_offset})"
        return repr_str
