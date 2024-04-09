# The MIT License (MIT)

# Copyright (c) 2024 Intel Corporation
# Copyright (c) 2019-2024, Mauricio Villegas <mauricio@omnius.com>

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
#
# NOTE: This code contains adaptations from code originally implemented in the
# jsonargparse project (https://github.com/omni-us/jsonargparse), which is
# licensed under the MIT License. However, this specific codebase is licensed
# under the Apache License 2.0. Please note this difference in licensing.
#
"""Jsonargparse functions to adapt OTX project."""

from functools import partial
import inspect
from typing import Any, Type
from jsonargparse._common import is_subclass
from jsonargparse._typehints import LazyInitBaseClass as _LazyInitBaseClass
from jsonargparse._util import ClassType

__all__ = ["lazy_instance", "ClassType"]


class LazyInitBaseClass(_LazyInitBaseClass):
    """Modifed LazyInitBaseClass to support callable classes.

    See https://github.com/omni-us/jsonargparse/issues/481 for more details.
    """

    def __init__(self, class_type: Type, lazy_kwargs: dict):
        self.__pickle_slot__ = {
            "class_type": class_type,
            "lazy_kwargs": lazy_kwargs,
        }

        self._lazy_call_method = None

        for name, member in inspect.getmembers(class_type, predicate=inspect.isfunction):
            if name == "__call__":
                self._lazy_call_method = partial(member, self)

        super().__init__(class_type=class_type, lazy_kwargs=lazy_kwargs)

    def __call__(self, *args, **kwargs):
        if self._lazy_call_method is None:
            return None

        self._lazy_init()
        return self._lazy_call_method(*args, **kwargs)

    def __reduce__(self) -> str | tuple[Any, ...]:
        return self.__class__, tuple(self.__pickle_slot__.values())


def lazy_instance(class_type: Type[ClassType], **kwargs) -> ClassType:
    """Instantiates a lazy instance of the given type.

    By lazy it is meant that the __init__ is delayed unit the first time that a
    method of the instance is called. It also provides a `lazy_get_init_data` method
    useful for serializing.

    Args:
        class_type: The class to instantiate.
        **kwargs: Any keyword arguments to use for instantiation.
    """
    caller_module = inspect.getmodule(inspect.stack()[1][0])
    class_name = f"LazyInstance_{class_type.__name__}"
    if hasattr(caller_module, class_name):
        lazy_init_class = getattr(caller_module, class_name)
        assert is_subclass(lazy_init_class, LazyInitBaseClass) and is_subclass(lazy_init_class, class_type)
    else:
        lazy_init_class = type(
            class_name,
            (LazyInitBaseClass, class_type),
            {"__doc__": f"Class for lazy instances of {class_type}"},
        )
        if caller_module is not None:
            lazy_init_class.__module__ = getattr(caller_module, "__name__", __name__)
            setattr(caller_module, lazy_init_class.__qualname__, lazy_init_class)
    return lazy_init_class(class_type, kwargs)
