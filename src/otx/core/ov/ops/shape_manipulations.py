"""Shape-mainpulation-related modules for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field

import torch

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.op import Attribute, Operation
from otx.core.ov.ops.type_conversions import ConvertV0


@dataclass
class SqueezeV0Attribute(Attribute):
    """SqueezeV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class SqueezeV0(Operation[SqueezeV0Attribute]):
    """SqueezeV0 class."""

    TYPE = "Squeeze"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = SqueezeV0Attribute

    def forward(self, inputs, dims=None):
        """SqueezeV0's forward function."""
        if dims is None:
            return torch.squeeze(inputs)

        if dims.dim() == 0:
            dims = torch.unsqueeze(dims, 0)

        max_dim = inputs.dim()
        dims = dims.detach().cpu().tolist()
        for i, dim in enumerate(dims):
            if dim < 0:
                dims[i] = max_dim + dim

        output = inputs
        for dim in sorted(dims, reverse=True):
            output = torch.squeeze(output, dim)

        return output


@dataclass
class UnsqueezeV0Attribute(Attribute):
    """UnsqueezeV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class UnsqueezeV0(Operation[UnsqueezeV0Attribute]):
    """UnsqueezeV0 class."""

    TYPE = "Unsqueeze"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = UnsqueezeV0Attribute

    def forward(self, inputs, dims):
        """UnsqueezeV0's forward function."""
        if dims.dim() == 0:
            dims = torch.unsqueeze(dims, 0)

        max_dim = inputs.dim()
        dims = dims.detach().cpu().tolist()
        if len(dims) > 1:
            for i, dim in enumerate(dims):
                if dim < 0:
                    dims[i] = max_dim + dim

        output = inputs
        for dim in sorted(dims, reverse=True):
            output = torch.unsqueeze(output, dim)

        return output


@dataclass
class ReshapeV1Attribute(Attribute):
    """ReshapeV1Attribute class."""

    special_zero: bool


@OPS.register()
class ReshapeV1(Operation[ReshapeV1Attribute]):
    """ReshapeV1 class."""

    TYPE = "Reshape"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ReshapeV1Attribute

    def forward(self, inputs, shape):
        """ReshapeV1's forward function."""
        target_shape = shape.detach().cpu().tolist()
        origin_shape = list(inputs.shape)
        for i, (origin_dim, target_dim) in enumerate(zip(origin_shape, target_shape)):
            if target_dim == 0 and self.attrs.special_zero:
                target_shape[i] = origin_dim
            elif target_dim == -1:
                break
        for i, (origin_dim, target_dim) in enumerate(zip(origin_shape[::-1], target_shape[::-1])):
            if target_dim == 0 and self.attrs.special_zero:
                target_shape[i] = origin_dim
            elif target_dim == -1:
                break
        return torch.reshape(inputs, target_shape)


@dataclass
class ShapeOfV0Attribute(Attribute):
    """ShapeOfV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class ShapeOfV0(Operation[ShapeOfV0Attribute]):
    """ShapeOfV0 class."""

    TYPE = "ShapeOf"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ShapeOfV0Attribute

    def forward(self, inputs):
        """ShapeOfV0's forward function."""
        return torch.tensor(inputs.shape, device=inputs.device)


@dataclass
class ShapeOfV3Attribute(Attribute):
    """ShapeOfV3Attribute class."""

    output_type: str = field(default="i64")

    def __post_init__(self):
        """ShapeOfV3Attribute's post-init function."""
        super().__post_init__()
        valid_output_type = ["i64", "i32"]
        if self.output_type not in valid_output_type:
            raise ValueError(f"Invalid output_type {self.output_type}. " f"It must be one of {valid_output_type}.")


@OPS.register()
class ShapeOfV3(Operation[ShapeOfV3Attribute]):
    """ShapeOfV3 class."""

    TYPE = "ShapeOf"
    VERSION = "opset3"
    ATTRIBUTE_FACTORY = ShapeOfV3Attribute

    def forward(self, inputs):
        """ShapeOfV3's forward function."""
        return ConvertV0("temp", shape=self.shape, destination_type=self.attrs.output_type)(
            torch.tensor(inputs.shape, device=inputs.device)
        )
