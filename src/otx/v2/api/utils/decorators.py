"""Decorator functions for API."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import TYPE_CHECKING, Callable, List, TypeVar, Union

if TYPE_CHECKING:  # pragma: no cover
    from otx.v2.api.core.auto_runner import AutoRunner
    from otx.v2.api.core.dataset import BaseDataset


def add_subset_dataloader(subsets: List[str]) -> Callable:
    """Decorate that adds dataloader methods for each subset in the given list of subsets.

    Args:
        subsets (List[str]): A list of subset names for which dataloader methods will be added.

    Returns:
        Callable: A decorator function that adds dataloader methods to a class.

    Example:
        >>> @add_subset_dataloader(["train", "val"])
            class MyDataset(BaseDataset):
        >>> MyDataset().train_dataloader()
        Dataloader()
    """

    def decorator(cls: Union["BaseDataset", "AutoRunner"]) -> Union["BaseDataset", "AutoRunner"]:
        def dataloader_method(subset: str) -> Callable:
            def wrapper(self, *args, **kwargs) -> TypeVar:  # noqa: ANN001
                kwargs["subset"] = subset
                return cls.subset_dataloader(self, *args, **kwargs)

            return wrapper

        if not hasattr(cls, "subset_dataloader"):
            msg = "In order to use this decorator, the class must have the subset_dataloader function implemented."
            raise NotImplementedError(
                msg,
            )

        for subset in subsets:
            method_name = f"{subset}_dataloader"
            setattr(cls, method_name, dataloader_method(subset))
        return cls

    return decorator
