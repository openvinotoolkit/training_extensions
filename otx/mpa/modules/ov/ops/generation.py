# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass

import torch

from .builder import OPS
from .op import Attribute, Operation
from .type_conversions import _ov_to_torch


@dataclass
class RangeV4Attribute(Attribute):
    output_type: str


@OPS.register()
class RangeV4(Operation[RangeV4Attribute]):
    TYPE = "Range"
    VERSION = 4
    ATTRIBUTE_FACTORY = RangeV4Attribute

    def forward(self, start, stop, step):
        dtype = _ov_to_torch[self.attrs.output_type]
        return torch.arange(
            start=start,
            end=stop,
            step=step,
            dtype=dtype,
            device=start.device,
            requires_grad=False,
        )
