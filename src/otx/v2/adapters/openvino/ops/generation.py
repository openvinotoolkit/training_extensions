"""Generation-related module for otx.v2.adapters.openvino.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from otx.v2.adapters.openvino.ops.builder import OPS
from otx.v2.adapters.openvino.ops.op import Attribute, Operation
from otx.v2.adapters.openvino.ops.type_conversions import _ov_to_torch


@dataclass
class RangeV4Attribute(Attribute):
    """RangeV4Attribute class."""

    output_type: str


@OPS.register()
class RangeV4(Operation[RangeV4Attribute]):
    """RangeV4 class."""

    TYPE = "Range"
    VERSION = 4
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
