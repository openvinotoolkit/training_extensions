# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field
from typing import Callable, List

import torch
from torch.nn import functional as F

from .builder import OPS
from .op import Attribute, Operation
from .utils import get_torch_padding


@dataclass
class ConvolutionV1Attribute(Attribute):
    strides: List[int]
    pads_begin: List[int]
    pads_end: List[int]
    dilations: List[int]
    auto_pad: str = field(default="explicit")

    def __post_init__(self):
        super().__post_init__()
        valid_auto_pad = ["explicit", "same_upper", "same_Lower", "valid"]
        if self.auto_pad not in valid_auto_pad:
            raise ValueError(f"Invalid auto_pad {self.auto_pad}. " f"It must be one of {valid_auto_pad}.")


@OPS.register()
class ConvolutionV1(Operation[ConvolutionV1Attribute]):
    TYPE = "Convolution"
    VERSION = 1
    ATTRIBUTE_FACTORY = ConvolutionV1Attribute

    def forward(self, input, weight):
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
            list(input.shape[2:]),
            list(weight.shape[2:]),
            self.attrs.strides,
            self.attrs.dilations,
        )
        if isinstance(padding, Callable):
            input = padding(input=input)
            padding = 0

        return func(
            input=input,
            weight=weight,
            bias=None,
            stride=self.attrs.strides,
            padding=padding,
            dilation=self.attrs.dilations,
        )


@dataclass
class GroupConvolutionV1Attribute(ConvolutionV1Attribute):
    pass


@OPS.register()
class GroupConvolutionV1(Operation[GroupConvolutionV1Attribute]):
    TYPE = "GroupConvolution"
    VERSION = 1
    ATTRIBUTE_FACTORY = GroupConvolutionV1Attribute

    def forward(self, input, weight):
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
            list(input.shape[2:]),
            list(weight.shape[2:]),
            self.attrs.strides,
            self.attrs.dilations,
        )
        if isinstance(padding, Callable):
            input = padding(input=input)
            padding = 0

        output = func(
            input=input,
            weight=weight,
            bias=None,
            stride=self.attrs.strides,
            padding=padding,
            dilation=self.attrs.dilations,
            groups=n_groups,
        )

        return output
