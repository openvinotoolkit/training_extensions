"""Decorator functions for API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, List


def add_subset_dataloader(subsets: List[str]) -> Callable:
    def decorator(cls) -> object:  # noqa: ANN001
        def dataloader_method(subset: str) -> Callable:
            def wrapper(self, *args, **kwargs) -> object:  # noqa: ANN001
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
