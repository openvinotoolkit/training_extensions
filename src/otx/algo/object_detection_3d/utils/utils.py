# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""utils for object detection 3D models."""
from __future__ import annotations

import torch
from torch import Tensor


# TODO(Kirill): try to remove this class
class NestedTensor:
    """Nested tensor class for object detection 3D models."""

    def __init__(self, tensors: Tensor, mask: Tensor) -> None:
        """Initialize a NestedTensor object.

        Args:
            tensors (Tensor): The tensors representing the nested structure.
            mask (Tensor): The mask indicating the valid elements in the tensors.
        """
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> NestedTensor:
        """Move the NestedTensor object to the specified device.

        Args:
            device: The device to move the tensors to.

        Returns:
            NestedTensor: The NestedTensor object with tensors moved to the specified device.
        """
        cast_tensor = self.tensors.to(device)
        cast_mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> tuple[Tensor, Tensor]:
        """Decompose the NestedTensor object into its constituent tensors and masks."""
        return self.tensors, self.mask

    def __repr__(self) -> str:
        """Return a string representation of the NestedTensor object."""
        return str(self.tensors)


def box_cxcylrtb_to_xyxy(x: Tensor) -> Tensor:
    """Transform bbox from cxcylrtb to xyxy representation."""
    x_c, y_c, k, r, t, b = x.unbind(-1)
    bb = [(x_c - k), (y_c - t), (x_c + r), (y_c + b)]
    return torch.stack(bb, dim=-1)
