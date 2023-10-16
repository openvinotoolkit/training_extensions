"""Arithmetics-related codes for otx.v2.adapters.openvino.ops.arithmetics."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class MultiplyV1Attribute(Attribute):
    """MultiplyV1Attribute class."""

    auto_broadcast: str = field(default="numpy")


@OPS.register()
class MultiplyV1(Operation[MultiplyV1Attribute]):
    """MultiplyV1 class."""

    TYPE = "Multiply"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = MultiplyV1Attribute
    attrs: MultiplyV1Attribute

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        """MultiplyV1's forward function."""
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            return input_0 * input_1
        if broadcast == "numpy":
            return input_0 * input_1
        raise NotImplementedError


@dataclass
class DivideV1Attribute(Attribute):
    """DivideV1Attribute class."""

    m_pythondiv: bool = field(default=True)
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class DivideV1(Operation[DivideV1Attribute]):
    """DivideV1 class."""

    TYPE = "Divide"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = DivideV1Attribute
    attrs: DivideV1Attribute

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        """DivideV1's forward function."""
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            output = input_0 / input_1
        elif broadcast == "numpy":
            output = input_0 / input_1
        else:
            raise NotImplementedError

        non_integer_types = [torch.float16, torch.float32, torch.float64, torch.bool]
        if self.attrs.m_pythondiv and input_0.dtype not in non_integer_types and input_1.dtype not in non_integer_types:
            output = output.type(input_0.dtype)

        return output


@dataclass
class AddV1Attribute(Attribute):
    """AddV1Attribute class."""

    auto_broadcast: str = field(default="numpy")


@OPS.register()
class AddV1(Operation[AddV1Attribute]):
    """AddV1 class."""

    TYPE = "Add"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = AddV1Attribute
    attrs: AddV1Attribute

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        """AddV1's forward function."""
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            return input_0 + input_1
        if broadcast == "numpy":
            return input_0 + input_1
        raise NotImplementedError


@dataclass
class SubtractV1Attribute(Attribute):
    """SubtractV1Attribute class."""

    auto_broadcast: str = field(default="numpy")


@OPS.register()
class SubtractV1(Operation[SubtractV1Attribute]):
    """SubtractV1 class."""

    TYPE = "Subtract"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = SubtractV1Attribute
    attrs: SubtractV1Attribute

    def forward(self, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        """SubtractV1's forward function."""
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            return input_0 - input_1
        if broadcast == "numpy":
            return input_0 - input_1
        raise NotImplementedError


@dataclass
class TanV0Attribute(Attribute):
    """TanV0Attribute class."""


@OPS.register()
class TanV0(Operation[TanV0Attribute]):
    """TanV0 class."""

    TYPE = "Tan"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = TanV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """TanV0's forward function."""
        return torch.tan(inputs)
