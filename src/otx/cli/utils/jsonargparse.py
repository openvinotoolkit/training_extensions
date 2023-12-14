# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Functions related to jsonargparse."""
from __future__ import annotations

import ast
from typing import Any, TypeVar

import docstring_parser
from jsonargparse import Namespace, dict_to_namespace


def get_short_docstring(component: TypeVar) -> str | None:
    """Get the short description from the docstring.

    Args:
        component (object): The component to get the docstring from

    Returns:
        Optional[str]: The short description
    """
    if component.__doc__ is None:
        return None
    docstring = docstring_parser.parse(component.__doc__)
    return docstring.short_description


# [FIXME]: Overriding Namespce.update to match mmengine.Config (DictConfig | dict)
# and prevent int, float types from being converted to str
# https://github.com/omni-us/jsonargparse/issues/236
def update(self: Namespace, value: Any, key: str | None = None,  # noqa: ANN401
        only_unset: bool = False) -> Namespace:
    """Sets or replaces all items from the given nested namespace.

    Args:
        value: A namespace to update multiple values or other type to set in a single key.
        key: Branch key where to set the value. Required if value is not namespace.
        only_unset: Whether to only set the value if not set in namespace.
    """
    _dict_type = False
    if isinstance(value, dict):
        # Dict -> Nested Namespace for overriding
        _dict_type = True
        value = dict_to_namespace(value)
    if not isinstance(value, (Namespace, dict)):
        if not key:
            msg = 'Key is required if value not a Namespace.'
            raise KeyError(msg)
        if not only_unset or key not in self:
            if key not in self or value is not None:
                if isinstance(value, str) and value.isnumeric():
                    value = ast.literal_eval(value)
                self[key] = value
            elif value is None:
                del self[key]
    else:
        prefix = key + '.' if key else ''
        for _key, val in value.items():
            if not only_unset or prefix + _key not in self:
                self.update(val, prefix + _key)
    if _dict_type and key is not None:
        # Dict or Namespace -> Dict
        self[key] = dict_to_namespace(self[key]).as_dict()
    return self

Namespace.update = update
