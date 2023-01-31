# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class MultiplyV1Attribute(Attribute):
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class MultiplyV1(Operation[MultiplyV1Attribute]):
    TYPE = "Multiply"
    VERSION = 1
    ATTRIBUTE_FACTORY = MultiplyV1Attribute

    def forward(self, input_0, input_1):
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            assert input_0.shape == input_1.shape
            return input_0 * input_1
        elif broadcast == "numpy":
            return input_0 * input_1
        else:
            raise NotImplementedError


@dataclass
class DivideV1Attribute(Attribute):
    m_pythondiv: bool = field(default=True)
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class DivideV1(Operation[DivideV1Attribute]):
    TYPE = "Divide"
    VERSION = 1
    ATTRIBUTE_FACTORY = DivideV1Attribute

    def forward(self, input_0, input_1):
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            assert input_0.shape == input_1.shape
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
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class AddV1(Operation[AddV1Attribute]):
    TYPE = "Add"
    VERSION = 1
    ATTRIBUTE_FACTORY = AddV1Attribute

    def forward(self, input_0, input_1):
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            assert input_0.shape == input_1.shape
            return input_0 + input_1
        elif broadcast == "numpy":
            return input_0 + input_1
        else:
            raise NotImplementedError


@dataclass
class SubtractV1Attribute(Attribute):
    auto_broadcast: str = field(default="numpy")


@OPS.register()
class SubtractV1(Operation[SubtractV1Attribute]):
    TYPE = "Subtract"
    VERSION = 1
    ATTRIBUTE_FACTORY = SubtractV1Attribute

    def forward(self, input_0, input_1):
        broadcast = self.attrs.auto_broadcast

        if broadcast == "none":
            assert input_0.shape == input_1.shape
            return input_0 - input_1
        elif broadcast == "numpy":
            return input_0 - input_1
        else:
            raise NotImplementedError


@dataclass
class TanV0Attribute(Attribute):
    pass


@OPS.register()
class TanV0(Operation[TanV0Attribute]):
    TYPE = "Tan"
    VERSION = 0
    ATTRIBUTE_FACTORY = TanV0Attribute

    def forward(self, input):
        return torch.tan(input)
