"""Type-conversion-related modules for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import torch

from otx.core.ov.ops.builder import OPS
from otx.core.ov.ops.op import Attribute, Operation

_torch_to_ov = {
    torch.uint8: ["u1", "u4", "u8"],
    torch.int8: ["i4", "i8"],
    torch.int16: ["i16"],
    torch.int32: ["i32"],
    torch.int64: ["i64"],
    torch.float16: ["f16"],
    torch.float32: ["f32"],
    torch.bool: ["boolean"],
}

_ov_to_torch = {
    "u1": torch.uint8,  # no type in torch
    "u4": torch.uint8,  # no type in torch
    "u8": torch.uint8,
    "u32": torch.int32,  # no type in torch
    "u64": torch.int64,  # no type in torch
    "i4": torch.int8,  # no type in torch
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "f16": torch.float16,
    "f32": torch.float32,
    "boolean": torch.bool,
}


@dataclass
class ConvertV0Attribute(Attribute):
    """ConvertV0Attribute class."""

    destination_type: str


@OPS.register()
class ConvertV0(Operation[ConvertV0Attribute]):
    """ConvertV0 class."""

    TYPE = "Convert"
    VERSION = 0
    ATTRIBUTE_FACTORY = ConvertV0Attribute

    @staticmethod
    def convert_ov_type(ov_type):
        """ConvertV0's convert_ov_type function."""
        if ov_type not in _ov_to_torch:
            raise NotImplementedError
        return _ov_to_torch[ov_type]

    @staticmethod
    def convert_torch_type(torch_type):
        """ConvertV0's convert_torch_type function."""
        if torch_type not in _torch_to_ov:
            raise NotImplementedError
        return _torch_to_ov[torch_type][-1]

    def forward(self, inputs):
        """ConvertV0's forward function."""
        return inputs.type(self.convert_ov_type(self.attrs.destination_type))
