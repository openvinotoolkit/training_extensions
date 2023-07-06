"""Image Processings-related code for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import List

import numpy as np
from torch.nn import functional as F

from .builder import OPS
from .movements import PadV1
from .op import Attribute, Operation

# pylint: disable=too-many-instance-attributes, too-many-branches


@dataclass
class InterpolateV4Attribute(Attribute):
    """InterpolateV4Attribute class."""

    mode: str
    shape_calculation_mode: str
    coordinate_transformation_mode: str = field(default="half_pixel")
    nearest_mode: str = field(default="round_prefer_floor")
    antialias: bool = field(default=False)
    pads_begin: List[int] = field(default_factory=lambda: [0])
    pads_end: List[int] = field(default_factory=lambda: [0])
    cube_coeff: float = field(default=-0.75)

    def __post_init__(self):
        """InterpolateV4Attribute's post-init function."""
        super().__post_init__()
        valid_mode = ["nearest", "linear", "linear_onnx", "cubic"]
        if self.mode not in valid_mode:
            raise ValueError(f"Invalid mode {self.mode}. " f"It must be one of {valid_mode}.")
        valid_shape_calculation_mode = ["sizes", "scales"]
        if self.shape_calculation_mode not in valid_shape_calculation_mode:
            raise ValueError(
                f"Invalid shape_calculation_mode {self.shape_calculation_mode}. "
                f"It must be one of {valid_shape_calculation_mode}."
            )
        valid_coordinate_transformation_mode = [
            "half_pixel",
            "pytorch_half_pixel",
            "asymmetric",
            "tf_half_pixel_for_nn",
            "align_corners",
        ]
        if self.coordinate_transformation_mode not in valid_coordinate_transformation_mode:
            raise ValueError(
                f"Invalid coordinate_transformation_mode {self.coordinate_transformation_mode}. "
                f"It must be one of {valid_coordinate_transformation_mode}."
            )
        valid_nearest_mode = [
            "round_prefer_floor",
            "round_prefer_ceil",
            "floor",
            "ceil",
            "simple",
        ]
        if self.nearest_mode not in valid_nearest_mode:
            raise ValueError(f"Invalid nearest_mode {self.nearest_mode}. " f"It must be one of {valid_nearest_mode}.")


@OPS.register()
class InterpolateV4(Operation[InterpolateV4Attribute]):
    """InterpolateV4 class."""

    TYPE = "Interpolate"
    VERSION = 4
    ATTRIBUTE_FACTORY = InterpolateV4Attribute

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = PadV1("tmp", shape=self.shape, pad_mode="constant")

    def forward(self, inputs, sizes, scales, axes=None):
        """InterpolateV4's forward function."""
        # TODO list
        # - handle 'linear_onnx' mode
        # - coordinate_transformation_mode
        # - nearest_mode
        # - cube_coeff
        # - antialias

        if axes is None:
            axes = list(range(inputs.dim()))
        else:
            axes = axes.detach().cpu().tolist()

        output = self.pad(inputs, self.attrs.pads_begin, self.attrs.pads_end, 0)

        mode = self.attrs.mode
        if mode in ("linear", "linear_onnx"):
            align_corners = False
            if output.dim() == 3:
                pass
            elif output.dim() == 4:
                mode = "bilinear"
            elif output.dim() == 5:
                mode = "trilinear"
        elif mode == "cubic":
            align_corners = False
            if output.dim() == 3:
                raise NotImplementedError
            if output.dim() == 4:
                mode = "bicubic"
            elif output.dim() == 5:
                raise NotImplementedError
        elif mode == "nearest":
            align_corners = None
            pass  # pylint: disable=unnecessary-pass
        else:
            raise NotImplementedError

        if self.attrs.shape_calculation_mode == "sizes":
            sizes = sizes.detach().cpu().numpy()
            sizes = sizes[np.argsort(axes)].tolist()
            if output.dim() == len(sizes):
                sizes = sizes[2:]

            return F.interpolate(
                input=output,
                size=sizes,
                scale_factor=None,
                mode=mode,
                align_corners=align_corners,
            )
        scales = scales.detach().cpu().numpy()
        scales = scales[np.argsort(axes)].tolist()
        if output.dim() == len(scales):
            scales = scales[2:]

        return F.interpolate(
            input=output,
            size=None,
            scale_factor=scales,
            mode=mode,
            align_corners=align_corners,
        )
