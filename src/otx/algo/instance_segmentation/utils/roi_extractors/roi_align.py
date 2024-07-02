# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Implementation modified from mmdeploy.mmcv.ops.roi_align.py.

Reference : https://github.com/open-mmlab/mmdeploy/blob/v1.3.1/mmdeploy/mmcv/ops/roi_align.py
"""

from __future__ import annotations

import torch
from torch.autograd import Function
from torchvision.ops.roi_align import RoIAlign


class RoIAlignMMCV(Function):
    """Rewrite of mmdeploy/mmcv/ops/roi_align.py."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(g: torch.Graph, *args, **kwargs) -> torch.Tensor:  # noqa: ARG004
        """Dummy forward function."""
        return RoIAlignMMCV.origin_output

    @staticmethod
    def symbolic(
        g: torch.Graph,
        input: torch.Tensor,  # noqa: A002
        rois: torch.Tensor,
        output_size: tuple[int, int],
        spatial_scale: float,
        sampling_ratio: int,
        pool_mode: str,
        aligned: bool,
    ) -> torch.Graph:
        """Rewrite symbolic function for RoIAlign."""
        from torch.onnx.symbolic_opset9 import _cast_Long
        from torch.onnx.symbolic_opset11 import add, select

        batch_indices = _cast_Long(
            g,
            g.op(
                "Squeeze",
                select(g, rois, 1, g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))),
                axes_i=[1],
            ),
            False,
        )
        rois = select(g, rois, 1, g.op("Constant", value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        if aligned is True:
            rois = add(g, rois, g.op("Constant", value_t=torch.tensor([-0.5 / spatial_scale], dtype=torch.float)))
        return g.op(
            "RoiAlign",
            input,
            rois,
            batch_indices,
            output_height_i=output_size[0],
            output_width_i=output_size[1],
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=sampling_ratio,
            mode_s=pool_mode,
        )


class OTXRoIAlign(RoIAlign):
    """Rewrite of mmdeploy/mmcv/ops/roi_align.py."""

    def export(
        self,
        input: torch.Tensor,  # noqa: A002
        rois: torch.Tensor,
    ) -> torch.Tensor:
        """Export OTXRoIAlign."""
        state = torch._C._get_tracing_state()  # noqa: SLF001
        origin_output = self(input, rois)
        RoIAlignMMCV.origin_output = origin_output
        torch._C._set_tracing_state(state)  # noqa: SLF001

        output_size = self.output_size
        spatial_scale = self.spatial_scale
        sampling_ratio = self.sampling_ratio
        pool_mode = "avg"
        aligned = self.aligned

        return RoIAlignMMCV.apply(
            input,
            rois,
            output_size,
            spatial_scale,
            sampling_ratio,
            pool_mode,
            aligned,
        )
