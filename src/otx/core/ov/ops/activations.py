"""Activation-related modules for otx.core.ov.ops.activations."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.op import Attribute, Operation


@dataclass
class SoftMaxV0Attribute(Attribute):
    """SoftMaxV0Attribute class."""

    axis: int = field(default=1)


@OPS.register()
class SoftMaxV0(Operation[SoftMaxV0Attribute]):
    """SoftMaxV0 class."""

    TYPE = "Softmax"
    VERSION = 0
    ATTRIBUTE_FACTORY = SoftMaxV0Attribute

    def forward(self, inputs):
        """SoftMaxV0's forward function."""
        return F.softmax(input=inputs, dim=self.attrs.axis)


@dataclass
class SoftMaxV1Attribute(Attribute):
    """SoftMaxV1Attribute class."""

    axis: int = field(default=1)


@OPS.register()
class SoftMaxV1(Operation[SoftMaxV1Attribute]):
    """SoftMaxV1 class."""

    TYPE = "Softmax"
    VERSION = 1
    ATTRIBUTE_FACTORY = SoftMaxV1Attribute

    def forward(self, inputs):
        """SoftMaxV1's forward function."""
        return F.softmax(input=inputs, dim=self.attrs.axis)


@dataclass
class ReluV0Attribute(Attribute):
    """ReluV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class ReluV0(Operation[ReluV0Attribute]):
    """ReluV0 class."""

    TYPE = "Relu"
    VERSION = 0
    ATTRIBUTE_FACTORY = ReluV0Attribute

    def forward(self, inputs):
        """ReluV0's forward function."""
        return F.relu(inputs)


@dataclass
class SwishV4Attribute(Attribute):
    """SwishV4Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class SwishV4(Operation[SwishV4Attribute]):
    """SwishV4 class."""

    TYPE = "Swish"
    VERSION = 4
    ATTRIBUTE_FACTORY = SwishV4Attribute

    def forward(self, inputs, beta=1.0):
        """SwishV4's forward function."""
        return inputs * torch.sigmoid(inputs * beta)


@dataclass
class SigmoidV0Attribute(Attribute):
    """SigmoidV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class SigmoidV0(Operation[SigmoidV0Attribute]):
    """SigmoidV0 class."""

    TYPE = "Sigmoid"
    VERSION = 0
    ATTRIBUTE_FACTORY = SigmoidV0Attribute

    def forward(self, inputs):
        """SigmoidV0's forward function."""
        return torch.sigmoid(inputs)


@dataclass
class ClampV0Attribute(Attribute):
    """ClampV0Attribute class."""

    min: float
    max: float


@OPS.register()
class ClampV0(Operation[ClampV0Attribute]):
    """ClampV0 class."""

    TYPE = "Clamp"
    VERSION = 0
    ATTRIBUTE_FACTORY = ClampV0Attribute

    def forward(self, inputs):
        """ClampV0's forward function."""
        return inputs.clamp(min=self.attrs.min, max=self.attrs.max)


@dataclass
class PReluV0Attribute(Attribute):
    """PReluV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class PReluV0(Operation[PReluV0Attribute]):
    """PReluV0 class."""

    TYPE = "PRelu"
    VERSION = 0
    ATTRIBUTE_FACTORY = PReluV0Attribute

    def forward(self, inputs, slope):
        """PReluV0's forward function."""
        return F.prelu(input=inputs, weight=slope)


@dataclass
class TanhV0Attribute(Attribute):
    """TanhV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class TanhV0(Operation[TanhV0Attribute]):
    """TanhV0 class."""

    TYPE = "Tanh"
    VERSION = 0
    ATTRIBUTE_FACTORY = TanhV0Attribute

    def forward(self, inputs):
        """TanhV0's forward function."""
        return F.tanh(inputs)


@dataclass
class EluV0Attribute(Attribute):
    """EluV0Attribute class."""

    alpha: float


@OPS.register()
class EluV0(Operation[EluV0Attribute]):
    """EluV0 class."""

    TYPE = "Elu"
    VERSION = 0
    ATTRIBUTE_FACTORY = EluV0Attribute

    def forward(self, inputs):
        """EluV0's forward function."""
        return F.elu(input=inputs, alpha=self.attrs.alpha)


@dataclass
class SeluV0Attribute(Attribute):
    """SeluV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class SeluV0(Operation[SeluV0Attribute]):
    """SeluV0 class."""

    TYPE = "Selu"
    VERSION = 0
    ATTRIBUTE_FACTORY = SeluV0Attribute

    def forward(self, inputs, alpha, lambda_):
        """SeluV0's forward function."""
        return lambda_ * F.elu(input=inputs, alpha=alpha)


@dataclass
class MishV4Attribute(Attribute):
    """MishV4Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class MishV4(Operation[MishV4Attribute]):
    """MishV4 class."""

    TYPE = "Mish"
    VERSION = 4
    ATTRIBUTE_FACTORY = MishV4Attribute

    def forward(self, inputs):
        """MishV4's forward function."""
        # NOTE: pytorch 1.8.2 does not have mish function
        #  return F.mish(input=input)
        return inputs * F.tanh(F.softplus(inputs))


@dataclass
class HSwishV4Attribute(Attribute):
    """HSwishV4Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class HSwishV4(Operation[HSwishV4Attribute]):
    """HSwishV4 class."""

    TYPE = "HSwish"
    VERSION = 4
    ATTRIBUTE_FACTORY = HSwishV4Attribute

    def forward(self, inputs):
        """HSwishV4's forward function."""
        return F.hardswish(input=inputs)


@dataclass
class HSigmoidV5Attribute(Attribute):
    """HSigmoidV5Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class HSigmoidV5(Operation[HSigmoidV5Attribute]):
    """HSigmoidV5 class."""

    TYPE = "HSigmoid"
    VERSION = 5
    ATTRIBUTE_FACTORY = HSigmoidV5Attribute

    def forward(self, inputs):
        """HSigmoidV5's forward function."""
        return F.hardsigmoid(input=inputs)


@dataclass
class ExpV0Attribute(Attribute):
    """ExpV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class ExpV0(Operation[ExpV0Attribute]):
    """ExpV0 class."""

    TYPE = "Exp"
    VERSION = 0
    ATTRIBUTE_FACTORY = ExpV0Attribute

    def forward(self, inputs):
        """ExpV0's forward function."""
        return torch.exp(inputs)


@dataclass
class HardSigmoidV0Attribute(Attribute):
    """HardSigmoidV0Attribute class."""

    pass  # pylint: disable=unnecessary-pass


@OPS.register()
class HardSigmoidV0(Operation[HardSigmoidV0Attribute]):
    """HardSigmoidV0 class."""

    TYPE = "HardSigmoid"
    VERSION = 0
    ATTRIBUTE_FACTORY = HardSigmoidV0Attribute

    def forward(self, inputs, alpha, beta):
        """HardSigmoidV0's forward function."""
        return torch.maximum(
            torch.zeros_like(inputs),
            torch.minimum(torch.ones_like(inputs), inputs * alpha + beta),
        )


@dataclass
class GeluV7Attribute(Attribute):
    """GeluV7Attribute class."""

    approximation_mode: str = field(default="ERF")

    def __post_init__(self):
        """GeluV7Attribute's post init function."""
        super().__post_init__()
        valid_approximation_mode = ["ERF", "tanh"]
        if self.approximation_mode not in valid_approximation_mode:
            raise ValueError(
                f"Invalid approximation_mode {self.approximation_mode}. "
                f"It must be one of {valid_approximation_mode}."
            )


@OPS.register()
class GeluV7(Operation[GeluV7Attribute]):
    """GeluV7 class."""

    TYPE = "Gelu"
    VERSION = 7
    ATTRIBUTE_FACTORY = GeluV7Attribute

    def forward(self, inputs):
        """GeluV7's forward function."""
        mode = self.attrs.approximation_mode
        if mode == "ERF":
            return F.gelu(input=inputs)
        if mode == "tanh":
            return (
                inputs * 0.5 * (1 + F.tanh(torch.sqrt(2 / torch.tensor(math.pi)) * (inputs + 0.044715 * inputs**3)))
            )
        return None
