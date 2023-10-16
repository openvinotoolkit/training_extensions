"""Activation-related modules for otx.v2.adapters.openvino.ops.activations."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch.nn import functional

from .builder import OPS
from .op import Attribute, Operation


@dataclass
class SoftMaxV0Attribute(Attribute):
    """SoftMaxV0Attribute class."""

    axis: int = field(default=1)


@OPS.register()
class SoftMaxV0(Operation[SoftMaxV0Attribute]):
    """SoftMaxV0 class."""

    TYPE = "Softmax"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = SoftMaxV0Attribute
    attrs: SoftMaxV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """SoftMaxV0's forward function."""
        return functional.softmax(input=inputs, dim=self.attrs.axis)


@dataclass
class SoftMaxV1Attribute(Attribute):
    """SoftMaxV1Attribute class."""

    axis: int = field(default=1)


@OPS.register()
class SoftMaxV1(Operation[SoftMaxV1Attribute]):
    """SoftMaxV1 class."""

    TYPE = "Softmax"
    VERSION = "opset8"
    ATTRIBUTE_FACTORY = SoftMaxV1Attribute
    attrs: SoftMaxV1Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """SoftMaxV1's forward function."""
        return functional.softmax(input=inputs, dim=self.attrs.axis)


@dataclass
class ReluV0Attribute(Attribute):
    """ReluV0Attribute class."""


@OPS.register()
class ReluV0(Operation[ReluV0Attribute]):
    """ReluV0 class."""

    TYPE = "Relu"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ReluV0Attribute
    attrs: ReluV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ReluV0's forward function."""
        return functional.relu(inputs)


@dataclass
class SwishV4Attribute(Attribute):
    """SwishV4Attribute class."""


@OPS.register()
class SwishV4(Operation[SwishV4Attribute]):
    """SwishV4 class."""

    TYPE = "Swish"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = SwishV4Attribute
    attrs: SwishV4Attribute

    def forward(self, inputs: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """SwishV4's forward function."""
        return inputs * torch.sigmoid(inputs * beta)


@dataclass
class SigmoidV0Attribute(Attribute):
    """SigmoidV0Attribute class."""


@OPS.register()
class SigmoidV0(Operation[SigmoidV0Attribute]):
    """SigmoidV0 class."""

    TYPE = "Sigmoid"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = SigmoidV0Attribute
    attrs: SigmoidV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
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
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ClampV0Attribute
    attrs: ClampV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ClampV0's forward function."""
        return inputs.clamp(min=self.attrs.min, max=self.attrs.max)


@dataclass
class PReluV0Attribute(Attribute):
    """PReluV0Attribute class."""


@OPS.register()
class PReluV0(Operation[PReluV0Attribute]):
    """PReluV0 class."""

    TYPE = "PRelu"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = PReluV0Attribute
    attrs: PReluV0Attribute

    def forward(self, inputs: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
        """PReluV0's forward function."""
        return functional.prelu(input=inputs, weight=slope)


@dataclass
class TanhV0Attribute(Attribute):
    """TanhV0Attribute class."""


@OPS.register()
class TanhV0(Operation[TanhV0Attribute]):
    """TanhV0 class."""

    TYPE = "Tanh"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = TanhV0Attribute
    attrs: TanhV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """TanhV0's forward function."""
        return functional.tanh(inputs)


@dataclass
class EluV0Attribute(Attribute):
    """EluV0Attribute class."""

    alpha: float


@OPS.register()
class EluV0(Operation[EluV0Attribute]):
    """EluV0 class."""

    TYPE = "Elu"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = EluV0Attribute
    attrs: EluV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """EluV0's forward function."""
        return functional.elu(input=inputs, alpha=self.attrs.alpha)


@dataclass
class SeluV0Attribute(Attribute):
    """SeluV0Attribute class."""


@OPS.register()
class SeluV0(Operation[SeluV0Attribute]):
    """SeluV0 class."""

    TYPE = "Selu"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = SeluV0Attribute
    attrs: SeluV0Attribute

    def forward(self, inputs: torch.Tensor, alpha: float, lambda_: torch.Tensor) -> torch.Tensor:
        """SeluV0's forward function."""
        return lambda_ * functional.elu(input=inputs, alpha=alpha)


@dataclass
class MishV4Attribute(Attribute):
    """MishV4Attribute class."""


@OPS.register()
class MishV4(Operation[MishV4Attribute]):
    """MishV4 class."""

    TYPE = "Mish"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = MishV4Attribute
    attrs: MishV4Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """MishV4's forward function."""
        # NOTE: pytorch 1.8.2 does not have mish function
        return inputs * functional.tanh(functional.softplus(inputs))


@dataclass
class HSwishV4Attribute(Attribute):
    """HSwishV4Attribute class."""


@OPS.register()
class HSwishV4(Operation[HSwishV4Attribute]):
    """HSwishV4 class."""

    TYPE = "HSwish"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = HSwishV4Attribute
    attrs: HSwishV4Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """HSwishV4's forward function."""
        return functional.hardswish(input=inputs)


@dataclass
class HSigmoidV5Attribute(Attribute):
    """HSigmoidV5Attribute class."""


@OPS.register()
class HSigmoidV5(Operation[HSigmoidV5Attribute]):
    """HSigmoidV5 class."""

    TYPE = "HSigmoid"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = HSigmoidV5Attribute
    attrs: HSigmoidV5Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """HSigmoidV5's forward function."""
        return functional.hardsigmoid(input=inputs)


@dataclass
class ExpV0Attribute(Attribute):
    """ExpV0Attribute class."""


@OPS.register()
class ExpV0(Operation[ExpV0Attribute]):
    """ExpV0 class."""

    TYPE = "Exp"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = ExpV0Attribute
    attrs: ExpV0Attribute

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """ExpV0's forward function."""
        return torch.exp(inputs)


@dataclass
class HardSigmoidV0Attribute(Attribute):
    """HardSigmoidV0Attribute class."""


@OPS.register()
class HardSigmoidV0(Operation[HardSigmoidV0Attribute]):
    """HardSigmoidV0 class."""

    TYPE = "HardSigmoid"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = HardSigmoidV0Attribute
    attrs: HardSigmoidV0Attribute

    def forward(self, inputs: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        """HardSigmoidV0's forward function."""
        return torch.maximum(
            torch.zeros_like(inputs),
            torch.minimum(torch.ones_like(inputs), inputs * alpha + beta),
        )


@dataclass
class GeluV7Attribute(Attribute):
    """GeluV7Attribute class."""

    approximation_mode: str = field(default="ERF")

    def __post_init__(self) -> None:
        """GeluV7Attribute's post init function."""
        super().__post_init__()
        valid_approximation_mode = ["ERF", "tanh"]
        if self.approximation_mode not in valid_approximation_mode:
            raise ValueError(
                f"Invalid approximation_mode {self.approximation_mode}. "
                f"It must be one of {valid_approximation_mode}.",
            )


@OPS.register()
class GeluV7(Operation[GeluV7Attribute]):
    """GeluV7 class."""

    TYPE = "Gelu"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = GeluV7Attribute
    attrs: GeluV7Attribute

    def forward(self, inputs: torch.Tensor) -> Optional[Union[torch.Tensor, tuple]]:
        """GeluV7's forward function."""
        mode = self.attrs.approximation_mode
        if mode == "ERF":
            return functional.gelu(input=inputs)
        if mode == "tanh":
            return (
                inputs
                * 0.5
                * (1 + functional.tanh(torch.sqrt(2 / torch.tensor(math.pi)) * (inputs + 0.044715 * inputs**3)))
            )
        return None
