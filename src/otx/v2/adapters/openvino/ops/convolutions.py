"""Convolutions-related module for otx.v2.adapters.openvino.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Callable, List

from otx.v2.adapters.openvino.ops.builder import OPS
from otx.v2.adapters.openvino.ops.movements import get_torch_padding
from otx.v2.adapters.openvino.ops.op import Attribute, Operation
from torch.nn import functional as F


@dataclass
class ConvolutionV1Attribute(Attribute):
    """ConvolutionV1Attribute class."""

    strides: List[int]
    pads_begin: List[int]
    pads_end: List[int]
    dilations: List[int]
    auto_pad: str = field(default="explicit")

    def __post_init__(self):
        """ConvolutionV1Attribute's post-init function."""
        super().__post_init__()
        valid_auto_pad = ["explicit", "same_upper", "same_Lower", "valid"]
        if self.auto_pad not in valid_auto_pad:
            raise ValueError(f"Invalid auto_pad {self.auto_pad}. " f"It must be one of {valid_auto_pad}.")


@OPS.register()
class ConvolutionV1(Operation[ConvolutionV1Attribute]):
    """ConvolutionV1 class."""

    TYPE = "Convolution"
    VERSION = 1
    ATTRIBUTE_FACTORY = ConvolutionV1Attribute

    def forward(self, inputs, weight):
        """ConvolutionV1's forward function."""
        if weight.dim() == 3:
            func = F.conv1d
        elif weight.dim() == 4:
            func = F.conv2d
        elif weight.dim() == 5:
            func = F.conv3d
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
        if isinstance(padding, Callable):
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

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class GroupConvolutionV1(Operation[GroupConvolutionV1Attribute]):
    """GroupConvolutionV1 class."""

    TYPE = "GroupConvolution"
    VERSION = 1
    ATTRIBUTE_FACTORY = GroupConvolutionV1Attribute

    def forward(self, inputs, weight):
        """GroupConvolutionV1's forward function."""
        if weight.dim() == 4:
            func = F.conv1d
        elif weight.dim() == 5:
            func = F.conv2d
        elif weight.dim() == 6:
            func = F.conv3d
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
        if isinstance(padding, Callable):
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
