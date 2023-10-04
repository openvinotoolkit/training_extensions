"""OPS (OperationRegistry) module for otx.v2.adapters.openvino.ops.builder."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional, TypeVar, Union

from otx.v2.adapters.openvino.registry import Registry

from .op import Operation


class OperationRegistry(Registry):
    """OperationRegistry class."""

    def __init__(self, name: str, add_name_as_attr: bool = False) -> None:
        super().__init__(name, add_name_as_attr)
        self._registry_dict_by_type: dict = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """Register function from name."""

        def wrap(obj) -> TypeVar:  # noqa: ANN001
            layer_name = name
            if layer_name is None:
                layer_name = obj.__name__
            layer_type = obj.TYPE
            layer_version = obj.VERSION
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, layer_name)
            self._register(obj, layer_name, layer_type, layer_version)
            return obj

        return wrap

    def _register(self, obj: TypeVar, name: str, types: Optional[str] = None, version: Optional[int] = None) -> None:
        """Register function from obj and obj name."""
        super()._register(obj, name, types, version)
        if types is None or version is None:
            return
        if types not in self._registry_dict_by_type:
            self._registry_dict_by_type[types] = {}
        if version in self._registry_dict_by_type[types]:
            raise KeyError(f"{version} is already registered in {types}")
        self._registry_dict_by_type[types][version] = obj

    def get_by_name(self, name: Union[str, TypeVar]) -> Union[str, TypeVar]:
        """Get obj from name."""
        return self.get(name)

    def get_by_type_version(self, types: str, version: str) -> Operation:
        """Get obj from type and version."""
        if types not in self._registry_dict_by_type:
            raise KeyError(f"type {types} is not registered in {self._name}")
        if version not in self._registry_dict_by_type[types]:
            raise KeyError(f"version {version} is not registered in {types} of {self._name}")
        return self._registry_dict_by_type[types][version]


OPS: OperationRegistry = OperationRegistry("ov ops")
