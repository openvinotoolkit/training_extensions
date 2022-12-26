# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class SoftMaxV0Attribute(Attribute):
    axis: int = field(default=1)


@OPS.register()
class SoftMaxV0(Operation[SoftMaxV0Attribute]):
    TYPE = "Softmax"
    VERSION = 0
    ATTRIBUTE_FACTORY = SoftMaxV0Attribute

    def forward(self, input):
        return F.softmax(input=input, dim=self.attrs.axis)


@dataclass
class SoftMaxV1Attribute(Attribute):
    axis: int = field(default=1)


@OPS.register()
class SoftMaxV1(Operation[SoftMaxV1Attribute]):
    TYPE = "Softmax"
    VERSION = 1
    ATTRIBUTE_FACTORY = SoftMaxV1Attribute

    def forward(self, input):
        return F.softmax(input=input, dim=self.attrs.axis)


@dataclass
class ReluV0Attribute(Attribute):
    pass


@OPS.register()
class ReluV0(Operation[ReluV0Attribute]):
    TYPE = "Relu"
    VERSION = 0
    ATTRIBUTE_FACTORY = ReluV0Attribute

    def forward(self, input):
        return F.relu(input)


@dataclass
class SwishV4Attribute(Attribute):
    pass


@OPS.register()
class SwishV4(Operation[SwishV4Attribute]):
    TYPE = "Swish"
    VERSION = 4
    ATTRIBUTE_FACTORY = SwishV4Attribute

    def forward(self, input, beta=1.0):
        return input * torch.sigmoid(input * beta)


@dataclass
class SigmoidV0Attribute(Attribute):
    pass


@OPS.register()
class SigmoidV0(Operation[SigmoidV0Attribute]):
    TYPE = "Sigmoid"
    VERSION = 0
    ATTRIBUTE_FACTORY = SigmoidV0Attribute

    def forward(self, input):
        return torch.sigmoid(input)


@dataclass
class ClampV0Attribute(Attribute):
    min: float
    max: float


@OPS.register()
class ClampV0(Operation[ClampV0Attribute]):
    TYPE = "Clamp"
    VERSION = 0
    ATTRIBUTE_FACTORY = ClampV0Attribute

    def forward(self, input):
        return input.clamp(min=self.attrs.min, max=self.attrs.max)


@dataclass
class PReluV0Attribute(Attribute):
    pass


@OPS.register()
class PReluV0(Operation[PReluV0Attribute]):
    TYPE = "PRelu"
    VERSION = 0
    ATTRIBUTE_FACTORY = PReluV0Attribute

    def forward(self, input, slope):
        return F.prelu(input=input, weight=slope)


@dataclass
class TanhV0Attribute(Attribute):
    pass


@OPS.register()
class TanhV0(Operation[TanhV0Attribute]):
    TYPE = "Tanh"
    VERSION = 0
    ATTRIBUTE_FACTORY = TanhV0Attribute

    def forward(self, input):
        return F.tanh(input)


@dataclass
class EluV0Attribute(Attribute):
    alpha: float


@OPS.register()
class EluV0(Operation[EluV0Attribute]):
    TYPE = "Elu"
    VERSION = 0
    ATTRIBUTE_FACTORY = EluV0Attribute

    def forward(self, input):
        return F.elu(input=input, alpha=self.attrs.alpha)


@dataclass
class SeluV0Attribute(Attribute):
    pass


@OPS.register()
class SeluV0(Operation[SeluV0Attribute]):
    TYPE = "Selu"
    VERSION = 0
    ATTRIBUTE_FACTORY = SeluV0Attribute

    def forward(self, input, alpha, lambda_):
        return lambda_ * F.elu(input=input, alpha=alpha)


@dataclass
class MishV4Attribute(Attribute):
    pass


@OPS.register()
class MishV4(Operation[MishV4Attribute]):
    TYPE = "Mish"
    VERSION = 4
    ATTRIBUTE_FACTORY = MishV4Attribute

    def forward(self, input):
        # NOTE: pytorch 1.8.2 does not have mish function
        #  return F.mish(input=input)
        return input * F.tanh(F.softplus(input))


@dataclass
class HSwishV4Attribute(Attribute):
    pass


@OPS.register()
class HSwishV4(Operation[HSwishV4Attribute]):
    TYPE = "HSwish"
    VERSION = 4
    ATTRIBUTE_FACTORY = HSwishV4Attribute

    def forward(self, input):
        return F.hardswish(input=input)


@dataclass
class HSigmoidV5Attribute(Attribute):
    pass


@OPS.register()
class HSigmoidV5(Operation[HSigmoidV5Attribute]):
    TYPE = "HSigmoid"
    VERSION = 5
    ATTRIBUTE_FACTORY = HSigmoidV5Attribute

    def forward(self, input):
        return F.hardsigmoid(input=input)


@dataclass
class ExpV0Attribute(Attribute):
    pass


@OPS.register()
class ExpV0(Operation[ExpV0Attribute]):
    TYPE = "Exp"
    VERSION = 0
    ATTRIBUTE_FACTORY = ExpV0Attribute

    def forward(self, input):
        return torch.exp(input)


@dataclass
class HardSigmoidV0Attribute(Attribute):
    pass


@OPS.register()
class HardSigmoidV0(Operation[HardSigmoidV0Attribute]):
    TYPE = "HardSigmoid"
    VERSION = 0
    ATTRIBUTE_FACTORY = HardSigmoidV0Attribute

    def forward(self, input, alpha, beta):
        return torch.maximum(
            torch.zeros_like(input),
            torch.minimum(torch.ones_like(input), input * alpha + beta),
        )


@dataclass
class GeluV7Attribute(Attribute):
    approximation_mode: str = field(default="ERF")

    def __post_init__(self):
        super().__post_init__()
        valid_approximation_mode = ["ERF", "tanh"]
        if self.approximation_mode not in valid_approximation_mode:
            raise ValueError(
                f"Invalid approximation_mode {self.approximation_mode}. "
                f"It must be one of {valid_approximation_mode}."
            )


@OPS.register()
class GeluV7(Operation[GeluV7Attribute]):
    TYPE = "Gelu"
    VERSION = 7
    ATTRIBUTE_FACTORY = GeluV7Attribute

    def forward(self, input):
        mode = self.attrs.approximation_mode
        if mode == "ERF":
            return F.gelu(input=input)
        elif mode == "tanh":
            return input * 0.5 * (1 + F.tanh(torch.sqrt(2 / torch.tensor(math.pi)) * (input + 0.044715 * input**3)))
