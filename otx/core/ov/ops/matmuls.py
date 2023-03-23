"""MatMul-related modules for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field

import torch

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.op import Attribute, Operation


@dataclass
class MatMulV0Attribute(Attribute):
    """MatMulV0Attribute class."""

    transpose_a: bool = field(default=False)
    transpose_b: bool = field(default=False)


@OPS.register()
class MatMulV0(Operation[MatMulV0Attribute]):
    """MatMulV0 class."""

    TYPE = "MatMul"
    VERSION = 0
    ATTRIBUTE_FACTORY = MatMulV0Attribute

    def forward(self, input_a, input_b):
        """MatMulV0's forward function."""
        if self.attrs.transpose_a:
            input_a = torch.transpose(input_a, -1, -2)
        if self.attrs.transpose_b:
            input_b = torch.transpose(input_b, -1, -2)
        return torch.matmul(input_a, input_b)


@dataclass
class EinsumV7Attribute(Attribute):
    """EinsumV7Attribute class."""

    equation: str


@OPS.register()
class EinsumV7(Operation[EinsumV7Attribute]):
    """EinsumV7 class."""

    TYPE = "Einsum"
    VERSION = 7
    ATTRIBUTE_FACTORY = EinsumV7Attribute

    def forward(self, *inputs):
        """EinsumV7's forward function."""
        return torch.einsum(self.attrs.equation, *inputs)
