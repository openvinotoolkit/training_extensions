# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for OTX data entities."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from otx.core.data.entity.base import T_OTXDataEntity


def register_pytree_node(cls: type[T_OTXDataEntity]) -> type[T_OTXDataEntity]:
    """Decorator to register an OTX data entity with PyTorch's PyTree.

    This decorator should be applied to every OTX data entity, as TorchVision V2 transforms
    use the PyTree to flatten and unflatten the data entity during runtime.

    Example:
        `MulticlassClsDataEntity` example ::

            @register_pytree_node
            @dataclass
            class MulticlassClsDataEntity(OTXDataEntity):
                ...
    """
    flatten_fn = lambda obj: (list(obj.values()), list(obj.keys()))  # noqa: E731
    unflatten_fn = lambda values, context: cls(**dict(zip(context, values)))  # noqa: E731
    pytree._register_pytree_node(  # noqa: SLF001
        typ=cls,
        flatten_fn=flatten_fn,
        unflatten_fn=unflatten_fn,
    )
    return cls
