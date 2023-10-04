"""Convolutions-related module for otx.v2.adapters.openvino.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import List

import torch
from torch.nn import functional

from .builder import OPS
from .movements import get_torch_padding
from .op import Attribute, Operation


@dataclass
class ConvolutionV1Attribute(Attribute):
    """ConvolutionV1Attribute class."""

    strides: List[int]
    pads_begin: List[int]
    pads_end: List[int]
    dilations: List[int]
    auto_pad: str = field(default="explicit")

    def __post_init__(self) -> None:
        """ConvolutionV1Attribute's post-init function."""
        super().__post_init__()
        valid_auto_pad = ["explicit", "same_upper", "same_Lower", "valid"]
        if self.auto_pad not in valid_auto_pad:
            raise ValueError(f"Invalid auto_pad {self.auto_pad}. " f"It must be one of {valid_auto_pad}.")


@OPS.register()
class ConvolutionV1(Operation[ConvolutionV1Attribute]):
    """ConvolutionV1 class."""

    TYPE = "Convolution"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ConvolutionV1Attribute
    attrs: ConvolutionV1Attribute

    def forward(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """ConvolutionV1's forward function."""
        if weight.dim() == 3:
            func = functional.conv1d
        elif weight.dim() == 4:
            func = functional.conv2d
        elif weight.dim() == 5:
            func = functional.conv3d
        else:
            raise NotImplementedError

        padding = get_torch_padding(
            self.attrs.pads_begin,
            self.attrs.pads_end,
            self.attrs.auto_pad,
            list(inputs.shape[2:]),
            list(weight.shape[2:]),
            self.attrs.strides,
            self.attrs.dilations,
        )
        if callable(padding):
            inputs = padding(input=inputs)
            padding = 0

        return func(
            input=inputs,
            weight=weight,
            bias=None,
            stride=self.attrs.strides,
            padding=padding,
            dilation=self.attrs.dilations,
        )


@dataclass
class GroupConvolutionV1Attribute(ConvolutionV1Attribute):
    """GroupConvolutionV1Attribute class."""

    # pylint: disable=unnecessary-pass


@OPS.register()
class GroupConvolutionV1(Operation[GroupConvolutionV1Attribute]):
    """GroupConvolutionV1 class."""

    TYPE = "GroupConvolution"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = GroupConvolutionV1Attribute
    attrs: GroupConvolutionV1Attribute

    def forward(self, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """GroupConvolutionV1's forward function."""
        if weight.dim() == 4:
            func = functional.conv1d
        elif weight.dim() == 5:
            func = functional.conv2d
        elif weight.dim() == 6:
            func = functional.conv3d
        else:
            raise NotImplementedError

        n_groups = weight.shape[0]
        # merge groups and out dimension
        weight = weight.view(-1, *weight.shape[2:])

        padding = get_torch_padding(
            self.attrs.pads_begin,
            self.attrs.pads_end,
            self.attrs.auto_pad,
            list(inputs.shape[2:]),
            list(weight.shape[2:]),
            self.attrs.strides,
            self.attrs.dilations,
        )
        if callable(padding):
            inputs = padding(input=inputs)
            padding = 0

        output = func(
            input=inputs,
            weight=weight,
            bias=None,
            stride=self.attrs.strides,
            padding=padding,
            dilation=self.attrs.dilations,
            groups=n_groups,
        )

        return output
