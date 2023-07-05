"""Custom Anchor Generator for SSD."""
# Copyright (C) 2018-2021 OpenMMLab
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import torch
from mmdet.core.anchor.anchor_generator import AnchorGenerator
from mmdet.core.anchor.builder import PRIOR_GENERATORS
from torch.nn.modules.utils import _pair

# TODO: Need to fix pylint issues
# pylint: disable=super-init-not-called, unused-argument


@PRIOR_GENERATORS.register_module()
class SSDAnchorGeneratorClustered(AnchorGenerator):
    """Custom Anchor Generator for SSD."""

    def __init__(self, strides, widths, heights, reclustering_anchors=False):
        self.strides = [_pair(stride) for stride in strides]
        self.widths = widths
        self.heights = heights
        self.centers = [(stride / 2.0, stride / 2.0) for stride in strides]

        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        """Generate base anchor for SSD."""
        multi_level_base_anchors = []
        for widths, heights, centers in zip(self.widths, self.heights, self.centers):
            base_anchors = self.gen_single_level_base_anchors(
                ws=torch.Tensor(widths), hs=torch.Tensor(heights), center=torch.Tensor(centers)
            )
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self, ws, hs, center):
        """Generate single_level_base_anchors for SSD."""
        x_center, y_center = center

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws, y_center + 0.5 * hs]
        base_anchors = torch.stack(base_anchors, dim=-1)

        return base_anchors

    def __repr__(self):
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
