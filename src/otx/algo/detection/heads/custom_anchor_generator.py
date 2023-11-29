"""Custom Anchor Generator for SSD."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from __future__ import annotations

import torch
from mmdet.models.task_modules.builder import PRIOR_GENERATORS
from mmdet.models.task_modules.prior_generators import AnchorGenerator
from torch.nn.modules.utils import _pair


@PRIOR_GENERATORS.register_module()
class SSDAnchorGeneratorClustered(AnchorGenerator):
    """Custom Anchor Generator for SSD."""

    def __init__(
        self,
        strides: tuple[int],
        widths: list[list[int]],
        heights: list[list[int]],
    ) -> None:
        """Initialize SSDAnchorGeneratorClustered.

        Args:
            strides (Tuple[int]): Anchor's strides.
            widths (List[List[int]]): Anchor's widths.
            heights (List[List[int]]): Anchor's height.
        """
        self.strides = [_pair(stride) for stride in strides]
        self.widths = widths
        self.heights = heights
        self.centers = [(stride / 2.0, stride / 2.0) for stride in strides]

        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()
        self.use_box_type = False

    def gen_base_anchors(self) -> list[torch.Tensor]:
        """Generate base anchor for SSD."""
        multi_level_base_anchors = []
        for widths, heights, centers in zip(self.widths, self.heights, self.centers):
            base_anchors = self.gen_single_level_base_anchors(
                widths=torch.Tensor(widths), heights=torch.Tensor(heights), center=torch.Tensor(centers),
            )
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def gen_single_level_base_anchors(
        self,
        widths:
        torch.Tensor,
        heights: torch.Tensor,
        center: torch.Tensor,
    ) -> torch.Tensor:
        """Generate single_level_base_anchors for SSD.

        Args:
            widths (torch.Tensor): Widths of base anchors.
            heights (torch.Tensor): Heights of base anchors.
            center (torch.Tensor): Centers of base anchors.
        """
        x_center, y_center = center

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * widths, y_center - 0.5 * heights, x_center + 0.5 * widths, y_center + 0.5 * heights,
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
