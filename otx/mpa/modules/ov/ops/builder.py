# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Any, Optional

from ..registry import Registry


class OperationRegistry(Registry):
    def __init__(self, name, add_name_as_attr=False):
        super().__init__(name, add_name_as_attr)
        self._registry_dict_by_type = {}

    def register(self, name: Optional[Any] = None):
        def wrap(obj):
            layer_name = name
            if layer_name is None:
                layer_name = obj.__name__
            layer_type = obj.TYPE
            layer_version = obj.VERSION
            assert layer_type != "" and layer_version >= 0
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, layer_name)
            self._register(obj, layer_name, layer_type, layer_version)
            return obj

        return wrap

    def _register(self, obj, name, type, version):
        super()._register(obj, name)
        if type not in self._registry_dict_by_type:
            self._registry_dict_by_type[type] = {}
        if version in self._registry_dict_by_type[type]:
            raise KeyError(f"{version} is already registered in {type}")
        self._registry_dict_by_type[type][version] = obj

    def get_by_name(self, name):
        return self.get(name)

    def get_by_type_version(self, type, version):
        if type not in self._registry_dict_by_type:
            raise KeyError(f"type {type} is not registered in {self._name}")
        if version not in self._registry_dict_by_type[type]:
            raise KeyError(f"version {version} is not registered in {type} of {self._name}")
        return self._registry_dict_by_type[type][version]


OPS = OperationRegistry("ov ops")
