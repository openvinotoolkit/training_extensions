"""Registry Class for otx.v2.adapters.openvino."""
# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Optional, TypeVar, Union


class Registry:
    """Registry Class for OMZ model."""

    REGISTERED_NAME_ATTR = "_registered_name"

    def __init__(self, name: str, add_name_as_attr: bool = False) -> None:
        self._name = name
        self._registry_dict: dict = {}
        self._add_name_as_attr = add_name_as_attr

    @property
    def registry_dict(self) -> dict:
        """Dictionary of registered module."""
        return self._registry_dict

    def _register(self, obj: TypeVar, name: str, types: Optional[str] = None, version: Optional[int] = None) -> None:
        """Register obj with name."""
        if name in self._registry_dict:
            raise KeyError(f"{name} is already registered in {self._name}")
        self._registry_dict[name] = obj

    def register(self, name: Optional[str] = None) -> Callable:
        """Register from name."""

        def wrap(obj, **kwargs) -> TypeVar:
            cls_name = name
            if cls_name is None:
                cls_name = obj.__name__
            if self._add_name_as_attr:
                setattr(obj, self.REGISTERED_NAME_ATTR, cls_name)
            self._register(obj, cls_name, **kwargs)
            return obj

        return wrap

    def get(self, key: Union[str, TypeVar]) -> Union[str, TypeVar]:
        """Get from module name (key)."""
        if key not in self._registry_dict:
            self._key_not_found(key)
        return self._registry_dict[key]

    def _key_not_found(self, key: Union[str, TypeVar]) -> KeyError:
        """Raise KeyError when key not founded."""
        raise KeyError(f"{key} is not found in {self._name}")

    def __contains__(self, item: Union[str, TypeVar]) -> bool:
        """Check containing of item."""
        return item in self._registry_dict.values()
