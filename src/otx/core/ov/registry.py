"""Registry Class for otx.core.ov."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional


class Registry:
    """Registry Class for OMZ model."""

    REGISTERED_NAME_ATTR = "_registered_name"

    def __init__(self, name, add_name_as_attr=False):
        self._name = name
        self._registry_dict = {}
        self._add_name_as_attr = add_name_as_attr

    @property
    def registry_dict(self) -> Dict[Any, Any]:
        """Dictionary of registered module."""
        return self._registry_dict

    def _register(self, obj: Any, name: Any):
        """Register obj with name."""
        if name in self._registry_dict:
            raise KeyError(f"{name} is already registered in {self._name}")
        self._registry_dict[name] = obj

    def register(self, name: Optional[Any] = None):
        """Register from name."""

        def wrap(obj):
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, cls_name)
            self._register(obj, cls_name)
            return obj

        return wrap

    def get(self, key: Any) -> Any:
        """Get from module name (key)."""
        if key not in self._registry_dict:
            self._key_not_found(key)
        return self._registry_dict[key]

    def _key_not_found(self, key: Any):
        """Raise KeyError when key not founded."""
        raise KeyError(f"{key} is not found in {self._name}")

    def __contains__(self, item):
        """Check containing of item."""
        return item in self._registry_dict.values()
