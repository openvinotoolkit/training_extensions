# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support TorchVision data transform functions."""

from __future__ import annotations

from inspect import isclass
from typing import TYPE_CHECKING

import hydra
import torch
import torchvision.transforms.v2 as tvt_v2

if TYPE_CHECKING:
    from numpy import array
    from torchvision.transforms.v2 import Compose

    from otx.core.config.data import SubsetConfig


class Identity(torch.nn.Module):
    """Indentity transform.

    This transform works as a placeholder and returns
    the same image. This is needed to run OpenVINO IR
    with OTX interface.
    """

    def forward(self, img: array) -> array:
        """Return the same img."""
        return img


class TorchVisionTransformLib:
    """Helper to support TorchVision transforms (only V2) in OTX."""

    @classmethod
    def list_available_transforms(cls) -> list[type[tvt_v2.Transform]]:
        """List available TorchVision transform (only V2) classes."""
        transforms = [
            obj
            for name in dir(tvt_v2)
            if (obj := getattr(tvt_v2, name)) and isclass(obj) and issubclass(obj, tvt_v2.Transform)
        ]
        transforms.append(Identity)
        return transforms

    @classmethod
    def generate(cls, config: SubsetConfig) -> Compose:
        """Generate TorchVision transforms from the configuration."""
        availables = set(cls.list_available_transforms())

        transforms = []
        return_as_list = False
        for cfg in config.transforms:
            transform = hydra.utils.instantiate(cfg)
            if isinstance(transform, Identity):
                # OV IR inference -> no augmentations
                return_as_list = True

            if type(transform) not in availables:
                msg = f"transform={transform} is not a valid TorchVision V2 transform"
                raise ValueError(msg)

            transforms.append(transform)

        if return_as_list:
            return transforms

        return tvt_v2.Compose(transforms)
