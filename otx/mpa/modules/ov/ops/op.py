# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import re
from dataclasses import dataclass, fields
from typing import Generic, Optional, Tuple, Type, TypeVar, Union

import torch

from ..utils import get_op_name
from .utils import get_dynamic_shape


@dataclass
class Attribute:
    shape: Optional[Union[Tuple[Tuple[int]], Tuple[int]]]

    def __post_init__(self):
        if self.shape is not None and not isinstance(self.shape, tuple):
            raise ValueError("shape must be a tuple of ints or a tuple of tuples of ints.")


_T = TypeVar("_T", bound=Attribute)


class Operation(torch.nn.Module, Generic[_T]):
    TYPE = ""
    VERSION = -1
    ATTRIBUTE_FACTORY: Type[_T] = Attribute

    def __init__(self, name: str, **kwargs):
        super().__init__()
        self._name = name
        self._attrs = self.ATTRIBUTE_FACTORY(**kwargs)

    @classmethod
    def from_ov(cls, ov_op):
        op_type = ov_op.get_type_name()
        op_version = ov_op.get_version()
        op_name = get_op_name(ov_op)
        assert cls.TYPE != "" and cls.VERSION >= 0
        assert op_type == cls.TYPE
        assert op_version == cls.VERSION

        attrs = ov_op.get_attributes()
        if "shape" not in attrs:
            shapes = []
            for output in ov_op.outputs():
                shapes.append(get_dynamic_shape(output))
            shapes = tuple(tuple(shape) for shape in shapes)
            attrs["shape"] = shapes

        return cls(name=op_name, **attrs)

    @property
    def type(self) -> str:
        return self.TYPE

    @property
    def version(self) -> int:
        return self.VERSION

    @property
    def name(self) -> str:
        return self._name

    @property
    def attrs(self):
        return self._attrs

    @property
    def shape(self) -> Optional[Union[Tuple[Tuple[int]], Tuple[int]]]:
        return self.attrs.shape
        #  shape = self.attrs.get("shape", None)
        #  if shape is not None and len(shape) == 1:
        #      shape = shape[0]
        #  return shape

    def __repr__(self):
        repr = f"{self.__class__.__name__}("
        repr += f"name={self.name}, "
        for field in fields(self.attrs):
            key = field.name
            if key == "shape":
                continue
            value = getattr(self.attrs, key)
            repr += f"{key}={value}, "
        repr = re.sub(", $", "", repr)
        repr += ")"
        return repr
