# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""utils for object detection 3D models."""

import torch
from torch import Tensor


class NestedTensor:
    def __init__(self, tensors: Tensor, mask: Tensor) -> None:
        """Initialize a NestedTensor object.

        Args:
            tensors (Tensor): The tensors representing the nested structure.
            mask (Tensor): The mask indicating the valid elements in the tensors.
        """
        self.tensors = tensors
        self.mask = mask

    def to(self, device: torch.device) -> "NestedTensor":
        """Move the NestedTensor object to the specified device.

        Args:
            device: The device to move the tensors to.

        Returns:
            NestedTensor: The NestedTensor object with tensors moved to the specified device.
        """
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self) -> tuple[Tensor, Tensor]:
        return self.tensors, self.mask

    def __repr__(self) -> str:
        return str(self.tensors)


def box_cxcylrtb_to_xyxy(x: Tensor) -> Tensor:
    """Transform bbox from cxcylrtb to xyxy representation."""
    x_c, y_c, l, r, t, b = x.unbind(-1)
    bb = [(x_c - l), (y_c - t), (x_c + r), (y_c + b)]
    return torch.stack(bb, dim=-1)
