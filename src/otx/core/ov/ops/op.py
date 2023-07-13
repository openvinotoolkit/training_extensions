"""Operation-related modules for otx.core.ov.ops."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

import re
from dataclasses import dataclass, fields
from typing import Generic, Optional, Tuple, Type, TypeVar, Union

import torch

from ..utils import get_op_name  # type: ignore[attr-defined]
from .utils import get_dynamic_shape


@dataclass
class Attribute:
    """Attribute class."""

    shape: Optional[Union[Tuple[Tuple[int]], Tuple[int]]]

    def __post_init__(self):
        """Attribute's post-init function."""
        if self.shape is not None and not isinstance(self.shape, tuple):
            raise ValueError("shape must be a tuple of ints or a tuple of tuples of ints.")


_T = TypeVar("_T", bound=Attribute)


class Operation(torch.nn.Module, Generic[_T]):  # pylint: disable=abstract-method, invalid-overridden-method
    """Operation class."""

    TYPE = ""
    VERSION = ""
    ATTRIBUTE_FACTORY: Type[Attribute] = Attribute

    def __init__(self, name: str, **kwargs):
        super().__init__()
        self._name = name
        self._attrs = self.ATTRIBUTE_FACTORY(**kwargs)

    @classmethod
    def from_ov(cls, ov_op):
        """Operation's from_ov function."""
        op_type = ov_op.get_type_name()
        op_version = ov_op.get_type_info().version_id
        op_name = get_op_name(ov_op)
        assert cls.TYPE and cls.VERSION
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
    def type(self) -> str:  # pylint: disable=invalid-overridden-method
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
    def attrs(self):
        """Operation's attrs property."""
        return self._attrs

    @property
    def shape(self) -> Optional[Union[Tuple[Tuple[int]], Tuple[int]]]:
        """Operation's shape property."""
        return self.attrs.shape

    def __repr__(self):
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
