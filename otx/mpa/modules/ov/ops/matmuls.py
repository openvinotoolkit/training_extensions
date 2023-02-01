# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field

import torch

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class MatMulV0Attribute(Attribute):
    transpose_a: bool = field(default=False)
    transpose_b: bool = field(default=False)


@OPS.register()
class MatMulV0(Operation[MatMulV0Attribute]):
    TYPE = "MatMul"
    VERSION = 0
    ATTRIBUTE_FACTORY = MatMulV0Attribute

    def forward(self, input_a, input_b):
        if self.attrs.transpose_a:
            input_a = torch.transpose(input_a, -1, -2)
        if self.attrs.transpose_b:
            input_b = torch.transpose(input_b, -1, -2)
        return torch.matmul(input_a, input_b)


@dataclass
class EinsumV7Attribute(Attribute):
    equation: str


@OPS.register()
class EinsumV7(Operation[EinsumV7Attribute]):
    TYPE = "Einsum"
    VERSION = 7
    ATTRIBUTE_FACTORY = EinsumV7Attribute

    def forward(self, *inputs):
        return torch.einsum(self.attrs.equation, *inputs)
