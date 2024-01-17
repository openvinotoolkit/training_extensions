# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

from inspect import isclass
from typing import TYPE_CHECKING

import torchvision.transforms.v2 as tvt_v2
from lightning.pytorch.cli import instantiate_class

if TYPE_CHECKING:
    from torchvision.transforms.v2 import Compose

    from otx.core.config.data import SubsetConfig


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
            transform = instantiate_class(args=(), init=cfg)
            if type(transform) not in availables:
                msg = f"transform={transform} is not a valid TorchVision V2 transform"
                raise ValueError(msg)

            transforms.append(transform)

        return tvt_v2.Compose(transforms)
