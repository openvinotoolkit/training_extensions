"""Operation-related modules for otx.v2.adapters.openvino.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass, fields
from typing import Generic, Optional, Type, TypeVar

import torch
from openvino.runtime import Node

from otx.v2.adapters.openvino.utils import get_op_name  # type: ignore[attr-defined]

from .utils import get_dynamic_shape


@dataclass
class Attribute:
    """Attribute class."""

    shape: Optional[tuple]

    def __post_init__(self) -> None:
        """Attribute's post-init function."""
        if self.shape is not None and not isinstance(self.shape, tuple):
            raise ValueError("shape must be a tuple of ints or a tuple of tuples of ints.")


_T = TypeVar("_T", bound=Attribute)


class Operation(torch.nn.Module, Generic[_T]):
    """Operation class."""

    TYPE = ""
    VERSION = ""
    ATTRIBUTE_FACTORY: Type[Attribute] = Attribute

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__()
        self._name = name
        self._attrs = self.ATTRIBUTE_FACTORY(**kwargs)

    @classmethod
    def from_ov(cls: torch.nn.Module, ov_op: Node) -> "Operation":
        """Operation's from_ov function."""
        op_name = get_op_name(ov_op)

        attrs = ov_op.get_attributes()
        if "shape" not in attrs:
            shapes = []
            for output in ov_op.outputs():
                shapes.append(get_dynamic_shape(output))
            attrs["shape"] = tuple(tuple(shape) for shape in shapes)

        return cls(name=op_name, **attrs)

    @property
    def type(self) -> str:
        """Operation's type property."""
        return self.TYPE

    @property
    def version(self) -> str:
        """Operation's version property."""
        return self.VERSION

    @property
    def name(self) -> str:
        """Operation's name property."""
        return self._name

    @property
    def attrs(self) -> Attribute:
        """Operation's attrs property."""
        return self._attrs

    @property
    def shape(self) -> Optional[tuple]:
        """Operation's shape property."""
        return self.attrs.shape

    def __repr__(self) -> str:
        """Operation's __repr__ function."""
        repr_str = f"{self.__class__.__name__}("
        repr_str += f"name={self.name}, "
        for field in fields(self.attrs):
            key = field.name
            if key == "shape":
                continue
            value = getattr(self.attrs, key)
            repr_str += f"{key}={value}, "
        repr_str = re.sub(", $", "", repr_str)
        repr_str += ")"
        return repr_str
