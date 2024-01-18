# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

from inspect import isclass
from typing import TYPE_CHECKING, Any

import hydra
import torch
import torchvision.transforms.v2 as tvt_v2
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F  # noqa: N812

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Compose

    from otx.core.config.data import SubsetConfig


class PerturbBoundingBoxes(tvt_v2.Transform):
    """Perturb bounding boxes with random offset value."""

    def __init__(self, offset: int) -> None:
        super().__init__()
        self.offset = offset

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # noqa: ANN401
        output = self._perturb_bounding_boxes(inpt, self.offset)
        return tv_tensors.wrap(output, like=inpt)

    def _perturb_bounding_boxes(self, inpt: torch.Tensor, offset: int) -> torch.Tensor:
        mean = torch.zeros_like(inpt)
        repeated_size = torch.tensor(inpt.canvas_size).repeat(len(inpt), 2)
        std = torch.minimum(repeated_size * 0.1, torch.tensor(offset))
        noise = torch.normal(mean, std)
        return (inpt + noise).clamp(mean, repeated_size - 1)


class PadtoSquare(tvt_v2.Transform):
    """Pad skewed image to square with zero padding."""

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:  # noqa: ANN401
        if isinstance(inpt, tv_tensors.BoundingBoxes):
            h, w = inpt.canvas_size
        elif isinstance(inpt, tv_tensors.Image):
            _, h, w = inpt.shape
        else:
            return inpt
        max_dim = max(w, h)
        pad_w = max_dim - w
        pad_h = max_dim - h
        padding = (0, 0, pad_w, pad_h)
        return self._call_kernel(F.pad, inpt, padding=padding, fill=0, padding_mode="constant")


tvt_v2.PerturbBoundingBoxes = PerturbBoundingBoxes
tvt_v2.PadtoSquare = PadtoSquare


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
