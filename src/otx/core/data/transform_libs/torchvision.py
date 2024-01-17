# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

from inspect import isclass
from typing import TYPE_CHECKING, Any, Dict, Tuple

from torch import Tensor
import torch
import hydra
import torchvision.transforms.v2 as tvt_v2
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import Image, BoundingBoxes

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Compose

    from otx.core.config.data import SubsetConfig
    
    
class PerturbBoundingBoxes(tvt_v2.Transform):
    """Perturb bounding boxes with random offset values."""
    def __init__(self, offset: int) -> None:
        super().__init__()
        self.offset = offset
        
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, BoundingBoxes):
            mean = torch.zeros_like(inpt)
            repeated_size = torch.tensor(inpt.canvas_size).repeat(len(inpt), 2)
            std = torch.minimum(repeated_size * 0.1, torch.tensor(self.offset))
            noise = torch.normal(mean, std)
            return (inpt + noise).clamp(mean, repeated_size)
        return inpt

class ResizewithLongestEdge(tvt_v2.Resize):
    """Resize images, masks, and bounding boxes to the longest edge."""
    def __init__(self, with_bbox: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_bbox = with_bbox

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, Image):
            return super()._transform(inpt, params)
        if self.with_bbox and isinstance(inpt, BoundingBoxes):
            return self.apply_coords(inpt.reshape(-1, 2, 2), inpt.canvas_size).reshape(-1, 4)
        return inpt
    
    def apply_coords(self, coords: Tensor, original_size: Tuple[int]) -> Tensor:
        """Expects torch tensor of length 2 in the final dimension.

        Args:
            coords (Tensor): Bounding boxes or points to be resized.

        Returns:
            Tensor: Resized coordinates.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, self.max_size)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords
    
    def get_preprocess_shape(self, oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """Compute the output size given input size and target long side length.

        Args:
            oldh (int): Original height.
            oldw (int): Original width.
            long_side_length (int): Target long side length.

        Returns:
            Tuple[int, int]: Output size.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class PadtoFixedSize(tvt_v2.Transform):
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if isinstance(inpt, (Image)):
            _, h, w = inpt.shape
            max_dim = max(w, h)
            pad_w = max_dim - w
            pad_h = max_dim - h
            padding = (0, 0, pad_w, pad_h)
            return self._call_kernel(F.pad, inpt, padding=padding, fill=0, padding_mode="constant")  # type: ignore[arg-type]
        return inpt


tvt_v2.PerturbBoundingBoxes = PerturbBoundingBoxes
tvt_v2.ResizewithLongestEdge = ResizewithLongestEdge
tvt_v2.PadtoFixedSize = PadtoFixedSize


class TorchVisionTransformLib:
    """Helper to support TorchVision transforms (only V2) in OTX."""

    @classmethod
    def list_available_transforms(cls) -> list[type[tvt_v2.Transform]]:
        """List available TorchVision transform (only V2) classes."""
        return [
            obj
            for name in dir(tvt_v2)
            if (obj := getattr(tvt_v2, name)) and isclass(obj) and issubclass(obj, tvt_v2.Transform)
        ]

    @classmethod
    def generate(cls, config: SubsetConfig) -> Compose:
        """Generate TorchVision transforms from the configuration."""
        availables = set(cls.list_available_transforms())

        transforms = []
        for cfg in config.transforms:
            transform = hydra.utils.instantiate(cfg)
            if type(transform) not in availables:
                msg = f"transform={transform} is not a valid TorchVision V2 transform"
                raise ValueError(msg)

            transforms.append(transform)

        return tvt_v2.Compose(transforms)
