# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for OTX data entities."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
import torchvision.transforms.v2.functional as F
from torch import Tensor
from torchvision import tv_tensors
from torchvision.utils import _log_api_usage_once

if TYPE_CHECKING:
    from otx.core.data.entity.base import T_OTXDataEntity, Points


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


def _clamp_points(points: Tensor, canvas_size: tuple[int, int]) -> Tensor:
    # TODO (sungchul): Tracking torchvision.transforms.v2.functional._meta._clamp_bounding_boxes
    # https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_meta.py#L234-L249
    in_dtype = points.dtype
    points = points.clone() if points.is_floating_point() else points.float()
    points[..., 0].clamp_(min=0, max=canvas_size[1])
    points[..., 1].clamp_(min=0, max=canvas_size[0])
    return points.to(in_dtype)


def clamp_points(inpt: Tensor, canvas_size: tuple[int, int] | None = None) -> Tensor:
    # TODO (sungchul): Tracking torchvision.transforms.v2.functional._meta.clamp_bounding_boxes
    # https://github.com/pytorch/vision/blob/main/torchvision/transforms/v2/functional/_meta.py#L252-L274
    if not torch.jit.is_scripting():
        _log_api_usage_once(clamp_points)

    if torch.jit.is_scripting() or F._utils.is_pure_tensor(inpt):
        if canvas_size is None:
            raise ValueError("For pure tensor inputs, `canvas_size` has to be passed.")
        return _clamp_points(inpt, canvas_size=canvas_size)
    elif isinstance(inpt, Points):
        if canvas_size is not None:
            raise ValueError("For point tv_tensor inputs, `canvas_size` must not be passed.")
        output = _clamp_points(inpt.as_subclass(Tensor), canvas_size=inpt.canvas_size)
        return tv_tensors.wrap(output, like=inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a point tv_tensor, but got {type(inpt)} instead."
        )
