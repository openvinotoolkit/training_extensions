"""OPS (OperationRegistry) module for otx.core.ov.ops.builder."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Any, Optional

from otx.core.ov.registry import Registry


class OperationRegistry(Registry):
    """OperationRegistry class."""

    def __init__(self, name, add_name_as_attr=False):
        super().__init__(name, add_name_as_attr)
        self._registry_dict_by_type = {}

    def register(self, name: Optional[Any] = None):
        """Register function from name."""

        def wrap(obj):
            layer_name = name
            if layer_name is None:
                layer_name = obj.__name__
            layer_type = obj.TYPE
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, layer_name)
            self._register(obj, layer_name, layer_type)
            return obj

        return wrap

    def _register(self, obj, name, types):
        """Register function from obj and obj name."""
        super()._register(obj, name)
        if types not in self._registry_dict_by_type:
            self._registry_dict_by_type[types] = obj

    def get_by_name(self, name):
        """Get obj from name."""
        return self.get(name)

    def get_by_type_version(self, types):
        """Get obj from type and version."""
        if types not in self._registry_dict_by_type:
            raise KeyError(f"type {types} is not registered in {self._name}")
        return self._registry_dict_by_type[types]


OPS = OperationRegistry("ov ops")
