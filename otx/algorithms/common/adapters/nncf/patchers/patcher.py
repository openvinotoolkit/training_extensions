# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import inspect
from functools import partial, partialmethod
from typing import Any, Callable, Optional


class Patcher:
    def __init__(self, wrapper: Callable, key: Any = "in_fn"):
        self._patched_modules = []
        self._wrapper = wrapper
        self._key = key

    def patch(
        self,
        obj_cls,
        fn_name: Optional[str] = None,
        wrapper: Optional[Callable] = None,
        key: Optional[Any] = None,
    ):
        assert ((wrapper == None) == (key == None)), "wrapper and key must be provided."

        if wrapper is None:
            wrapper = self._wrapper
            key = self._key

        obj_cls, fn_name = self.import_obj(obj_cls, fn_name)

        # wrap only if function does exist
        n_args = len(inspect.getargspec(obj_cls.__getattribute__)[0])
        if n_args == 1:
            try:
                fn = obj_cls.__getattribute__(fn_name)
            except AttributeError:
                return
            self._patch_module_fn(obj_cls, fn_name, fn, wrapper, key)
        else:
            if inspect.isclass(obj_cls):
                try:
                    fn = obj_cls.__getattribute__(obj_cls, fn_name)
                except AttributeError:
                    return
                self._patch_class_fn(obj_cls, fn_name, fn, wrapper, key)
            else:
                try:
                    fn = obj_cls.__getattribute__(fn_name)
                except AttributeError:
                    return
                self._patch_instance_fn(obj_cls, fn_name, fn, wrapper, key)

    def import_obj(self, obj_cls, fn_name=None):
        if fn_name is None:
            assert isinstance(obj_cls, str)
            fn_name = obj_cls.split(".")[-1]
            obj_cls = ".".join(obj_cls.split(".")[:-1])
        if isinstance(obj_cls, str):
            try:
                obj_cls = importlib.import_module(obj_cls)
            except ModuleNotFoundError:
                module = ".".join(obj_cls.split(".")[:-1])
                obj_cls = obj_cls.split(".")[-1]
                obj_cls = getattr(importlib.import_module(module), obj_cls)
        return obj_cls, fn_name

    def _patch_module_fn(self, obj_cls, fn_name, fn, wrapper, key):
        assert len(inspect.getargspec(obj_cls.__getattribute__)[0]) == 1
        obj_cls_path = obj_cls.__name__
        if (obj_cls_path, fn_name) not in self._patched_modules:
            setattr(obj_cls, fn_name, partial(wrapper, **{key: fn}))
            self._patched_modules.append((obj_cls_path, fn_name))

    def _patch_class_fn(self, obj_cls, fn_name, fn, wrapper, key):
        assert len(inspect.getargspec(obj_cls.__getattribute__)[0]) == 2
        obj_cls_path = obj_cls.__module__ + "." + obj_cls.__name__
        if (obj_cls_path, fn_name) not in self._patched_modules:
            setattr(
                obj_cls,
                fn_name,
                partialmethod(wrapper, **{key: fn}),
            )
            self._patched_modules.append((obj_cls_path, fn_name))

    def _patch_instance_fn(self, obj_cls, fn_name, fn, wrapper, key):
        assert len(inspect.getargspec(obj_cls.__getattribute__)[0]) == 2
        obj_cls_path = id(obj_cls)
        if (obj_cls_path, fn_name) not in self._patched_modules:
            setattr(
                obj_cls,
                fn_name,
                partialmethod(wrapper, **{key: fn}).__get__(obj_cls),
            )
            self._patched_modules.append((obj_cls_path, fn_name))
