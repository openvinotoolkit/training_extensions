# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass, field
from typing import Callable, List

from torch.nn import functional as F

from .builder import OPS
from .op import Attribute, Operation
from .utils import get_torch_padding


@dataclass
class MaxPoolV0Attribute(Attribute):
    strides: List[int]
    pads_begin: List[int]
    pads_end: List[int]
    kernel: List[int]
    rounding_type: str = field(default="floor")
    auto_pad: str = field(default="explicit")
    dilations: List[int] = field(default_factory=lambda: [])
    index_element_type: str = field(default="i64")
    axis: int = field(default=0)

    def __post_init__(self):
        super().__post_init__()
        valid_auto_pad = ["explicit", "same_upper", "same_Lower", "valid"]
        if self.auto_pad not in valid_auto_pad:
            raise ValueError(f"Invalid auto_pad {self.auto_pad}. " f"It must be one of {valid_auto_pad}.")
        valid_rounding_type = ["ceil", "floor"]
        if self.rounding_type not in valid_rounding_type:
            raise ValueError(
                f"Invalid rounding_type {self.rounding_type}. " f"It must be one of {valid_rounding_type}."
            )
        valid_index_element_type = ["i32", "i64"]
        if self.index_element_type not in valid_index_element_type:
            raise ValueError(
                f"Invalid index_element_type {self.index_element_type}. "
                f"It must be one of {valid_index_element_type}."
            )

        if not self.dilations:
            self.dilations = [1 for _ in self.strides]

        if self.axis != 0:
            raise NotImplementedError


@OPS.register()
class MaxPoolV0(Operation[MaxPoolV0Attribute]):
    TYPE = "MaxPool"
    VERSION = 0
    ATTRIBUTE_FACTORY = MaxPoolV0Attribute

    def forward(self, input):

        if input.dim() == 3:
            func = F.max_pool1d
        elif input.dim() == 4:
            func = F.max_pool2d
        elif input.dim() == 5:
            func = F.max_pool3d
        else:
            raise NotImplementedError

        padding = get_torch_padding(
            self.attrs.pads_begin,
            self.attrs.pads_end,
            self.attrs.auto_pad,
            list(input.shape[2:]),
            self.attrs.kernel,
            self.attrs.strides,
        )
        if isinstance(padding, Callable):
            input = padding(input=input)
            padding = 0

        return func(
            input=input,
            kernel_size=self.attrs.kernel,
            stride=self.attrs.strides,
            padding=padding,
            dilation=self.attrs.dilations,
            ceil_mode=self.attrs.rounding_type == "ceil",
            return_indices=True,
        )


@dataclass
class AvgPoolV1Attribute(Attribute):
    exclude_pad: bool
    strides: List[int]
    pads_begin: List[int]
    pads_end: List[int]
    kernel: List[int]
    rounding_type: str = field(default="floor")
    auto_pad: str = field(default="explicit")

    def __post_init__(self):
        super().__post_init__()
        valid_auto_pad = ["explicit", "same_upper", "same_Lower", "valid"]
        if self.auto_pad not in valid_auto_pad:
            raise ValueError(f"Invalid auto_pad {self.auto_pad}. " f"It must be one of {valid_auto_pad}.")
        valid_rounding_type = ["ceil", "floor"]
        if self.rounding_type not in valid_rounding_type:
            raise ValueError(
                f"Invalid rounding_type {self.rounding_type}. " f"It must be one of {valid_rounding_type}."
            )


@OPS.register()
class AvgPoolV1(Operation[AvgPoolV1Attribute]):
    TYPE = "AvgPool"
    VERSION = 1
    ATTRIBUTE_FACTORY = AvgPoolV1Attribute

    def __init__(self, *args, **kwargs):
        if "exclude-pad" in kwargs:
            kwargs["exclude_pad"] = kwargs.pop("exclude-pad")
        super().__init__(*args, **kwargs)

    def forward(self, input):
        if input.dim() == 3:
            func = F.avg_pool1d
        elif input.dim() == 4:
            func = F.avg_pool2d
        elif input.dim() == 5:
            func = F.avg_pool3d
        else:
            raise NotImplementedError

        padding = get_torch_padding(
            self.attrs.pads_begin,
            self.attrs.pads_end,
            self.attrs.auto_pad,
            list(input.shape[2:]),
            self.attrs.kernel,
            self.attrs.strides,
        )
        if isinstance(padding, Callable):
            input = padding(input=input)
            padding = 0

        return func(
            input=input,
            kernel_size=self.attrs.kernel,
            stride=self.attrs.strides,
            padding=padding,
            ceil_mode=self.attrs.rounding_type == "ceil",
            count_include_pad=not self.attrs.exclude_pad,
        )
