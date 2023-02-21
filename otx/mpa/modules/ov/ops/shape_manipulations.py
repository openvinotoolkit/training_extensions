# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation
from .type_conversions import ConvertV0


@dataclass
class SqueezeV0Attribute(Attribute):
    pass


@OPS.register()
class SqueezeV0(Operation[SqueezeV0Attribute]):
    TYPE = "Squeeze"
    VERSION = 0
    ATTRIBUTE_FACTORY = SqueezeV0Attribute

    def forward(self, input, dims=None):
        if dims is None:
            return torch.squeeze(input)

        if dims.dim() == 0:
            dims = torch.unsqueeze(dims, 0)

        max_dim = input.dim()
        dims = dims.detach().cpu().tolist()
        for i, dim in enumerate(dims):
            if dim < 0:
                dims[i] = max_dim + dim

        output = input
        for dim in sorted(dims, reverse=True):
            output = torch.squeeze(output, dim)

        return output


@dataclass
class UnsqueezeV0Attribute(Attribute):
    pass


@OPS.register()
class UnsqueezeV0(Operation[UnsqueezeV0Attribute]):
    TYPE = "Unsqueeze"
    VERSION = 0
    ATTRIBUTE_FACTORY = UnsqueezeV0Attribute

    def forward(self, input, dims):

        if dims.dim() == 0:
            dims = torch.unsqueeze(dims, 0)

        max_dim = input.dim()
        dims = dims.detach().cpu().tolist()
        if len(dims) > 1:
            for i, dim in enumerate(dims):
                if dim < 0:
                    dims[i] = max_dim + dim

        output = input
        for dim in sorted(dims, reverse=True):
            output = torch.unsqueeze(output, dim)

        return output


@dataclass
class ReshapeV1Attribute(Attribute):
    special_zero: bool


@OPS.register()
class ReshapeV1(Operation[ReshapeV1Attribute]):
    TYPE = "Reshape"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReshapeV1Attribute

    def forward(self, input, shape):
        target_shape = shape.detach().cpu().tolist()
        origin_shape = list(input.shape)
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
        return torch.reshape(input, target_shape)


@dataclass
class ShapeOfV0Attribute(Attribute):
    pass


@OPS.register()
class ShapeOfV0(Operation[ShapeOfV0Attribute]):
    TYPE = "ShapeOf"
    VERSION = 0
    ATTRIBUTE_FACTORY = ShapeOfV0Attribute

    def forward(self, input):
        return torch.tensor(input.shape, device=input.device)


@dataclass
class ShapeOfV3Attribute(Attribute):
    output_type: str = field(default="i64")

    def __post_init__(self):
        super().__post_init__()
        valid_output_type = ["i64", "i32"]
        if self.output_type not in valid_output_type:
            raise ValueError(f"Invalid output_type {self.output_type}. " f"It must be one of {valid_output_type}.")


@OPS.register()
class ShapeOfV3(Operation[ShapeOfV3Attribute]):
    TYPE = "ShapeOf"
    VERSION = 3
    ATTRIBUTE_FACTORY = ShapeOfV3Attribute

    def forward(self, input):
        return ConvertV0("temp", shape=self.shape, destination_type=self.attrs.output_type)(
            torch.tensor(input.shape, device=input.device)
        )
