"""Generation-related module for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import torch

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.op import Attribute, Operation
from otx.core.ov.ops.type_conversions import _ov_to_torch


@dataclass
class RangeV4Attribute(Attribute):
    """RangeV4Attribute class."""

    output_type: str


@OPS.register()
class RangeV4(Operation[RangeV4Attribute]):
    """RangeV4 class."""

    TYPE = "Range"
    VERSION = "opset1"
    ATTRIBUTE_FACTORY = RangeV4Attribute

    def forward(self, start, stop, step):
        """RangeV4's forward function."""
        dtype = _ov_to_torch[self.attrs.output_type]
        return torch.arange(
            start=start,
            end=stop,
            step=step,
            dtype=dtype,
            device=start.device,
            requires_grad=False,
        )
