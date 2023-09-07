"""Decorator functions for API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import inspect
from typing import Callable, List


def add_subset_dataloader(subsets: List[str]):
    def decorator(cls):
        def dataloader_method(subset: str):
            def wrapper(self, *args, **kwargs):
                return cls.subset_dataloader(self, subset=subset, *args, **kwargs)

            return wrapper

        if not hasattr(cls, "subset_dataloader"):
            raise NotImplementedError(
                "In order to use this decorator, the class must have the subset_dataloader function implemented."
            )

        for subset in subsets:
            method_name = f"{subset}_dataloader"
            setattr(cls, method_name, dataloader_method(subset))
        return cls

    return decorator


def set_default_argument(target_function: Callable):
    def decorator(func):
        # Get Trainer arguments
        target_args = inspect.signature(target_function).parameters

        # Add Trainer arguments to function
        for arg_name, arg_value in target_args.items():
            if arg_name not in func.__annotations__:
                func.__annotations__[arg_name] = arg_value.annotation

            if arg_name not in func.__code__.co_varnames:
                func.__code__ = func.__code__.replace(co_varnames=func.__code__.co_varnames + (arg_name,))

        # Update kwargs with default values from Trainer arguments
        def get_defaults(args):
            return {
                arg_name: arg_value.default
                for arg_name, arg_value in args.items()
                if arg_value.default != inspect.Parameter.empty
            }

        target_defaults = get_defaults(target_args)
        func_args = inspect.signature(func).parameters
        func_defaults = get_defaults(func_args)

        def wrapper(self, *args, **kwargs):
            updated_kwargs = target_defaults.copy()
            updated_kwargs.update(func_defaults)
            if args:
                updated_kwargs.update(args)
            updated_kwargs.update(kwargs)

            return func(self, **updated_kwargs)

        return wrapper

    return decorator
