"""Redunction-related modules for otx.v2.adapters.openvino.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class ReduceMeanV1Attribute(Attribute):
    """ReduceMeanV1Attribute class."""

    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceMeanV1(Operation[ReduceMeanV1Attribute]):
    """ReduceMeanV1 class."""

    TYPE = "ReduceMean"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceMeanV1Attribute
    attrs: ReduceMeanV1Attribute

    def forward(self, inputs: torch.Tensor, axes: Optional[Union[list, torch.Tensor]]) -> torch.Tensor:
        """ReduceMeanV1's forward function."""
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return inputs

        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        return torch.mean(input=inputs, dim=axes, keepdim=self.attrs.keep_dims)


@dataclass
class ReduceProdV1Attribute(Attribute):
    """ReduceMeanV1Attribute class."""

    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceProdV1(Operation[ReduceProdV1Attribute]):
    """ReduceMeanV1Attribute class."""

    TYPE = "ReduceProd"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceProdV1Attribute
    attrs: ReduceProdV1Attribute

    def forward(self, inputs: torch.Tensor, axes: Optional[Union[list, torch.Tensor]]) -> torch.Tensor:
        """ReduceMeanV1Attribute's forward function."""
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return inputs

        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        output = inputs
        for axe in axes:
            output = torch.prod(input=output, dim=axe, keepdim=True)
        if not self.attrs.keep_dims:
            output = torch.squeeze(output)

        return output


@dataclass
class ReduceMinV1Attribute(Attribute):
    """ReduceMinV1Attribute class."""

    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceMinV1(Operation[ReduceMinV1Attribute]):
    """ReduceMinV1 class."""

    TYPE = "ReduceMin"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceMinV1Attribute
    attrs: ReduceMinV1Attribute

    def forward(self, inputs: torch.Tensor, axes: Optional[Union[list, torch.Tensor]]) -> torch.Tensor:
        """ReduceMinV1's forward function."""
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return inputs

        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        output = inputs
        for axe in axes:
            output = torch.min(input=output, dim=axe, keepdim=True)[0]
        if not self.attrs.keep_dims:
            output = torch.squeeze(output)

        return output


@dataclass
class ReduceSumV1Attribute(Attribute):
    """ReduceSumV1Attribute class."""

    keep_dims: bool = field(default=False)


@OPS.register()
class ReduceSumV1(Operation[ReduceSumV1Attribute]):
    """ReduceSumV1 class."""

    TYPE = "ReduceSum"
    VERSION = 1
    ATTRIBUTE_FACTORY = ReduceSumV1Attribute
    attrs: ReduceSumV1Attribute

    def forward(self, inputs: torch.Tensor, axes: Optional[Union[list, torch.Tensor]]) -> torch.Tensor:
        """ReduceSumV1's forward function."""
        if isinstance(axes, torch.Tensor):
            axes = axes.tolist()
        if not axes:
            return inputs

        return torch.sum(input=inputs, dim=axes, keepdim=self.attrs.keep_dims)
