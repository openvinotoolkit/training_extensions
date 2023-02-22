# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class ReduceMeanV1Attribute(Attribute):
    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceMeanV1(Operation[ReduceMeanV1Attribute]):
    TYPE = "ReduceMean"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceMeanV1Attribute

    def forward(self, input, axes):
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return input

        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        return torch.mean(input=input, dim=axes, keepdim=self.attrs.keep_dims)


@dataclass
class ReduceProdV1Attribute(Attribute):
    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceProdV1(Operation[ReduceProdV1Attribute]):
    TYPE = "ReduceProd"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceProdV1Attribute

    def forward(self, input, axes):
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return input

        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        output = input
        for ax in axes:
            output = torch.prod(input=output, dim=ax, keepdim=True)
        if not self.attrs.keep_dims:
            output = torch.squeeze(output)

        return output


@dataclass
class ReduceMinV1Attribute(Attribute):
    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceMinV1(Operation[ReduceMinV1Attribute]):
    TYPE = "ReduceMin"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceMinV1Attribute

    def forward(self, input, axes):
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return input

        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        output = input
        for ax in axes:
            output = torch.min(input=output, dim=ax, keepdim=True)[0]
        if not self.attrs.keep_dims:
            output = torch.squeeze(output)

        return output


@dataclass
class ReduceSumV1Attribute(Attribute):
    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceSumV1(Operation[ReduceSumV1Attribute]):
    TYPE = "ReduceSum"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceSumV1Attribute

    def forward(self, input, axes):
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return input

        return torch.sum(input=input, dim=axes, keepdim=self.attrs.keep_dims)
