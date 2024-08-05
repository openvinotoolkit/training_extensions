# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Helper to support MMCV data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from mmcv.transforms import LoadImageFromFile as MMCVLoadImageFromFile
from mmcv.transforms.builder import TRANSFORMS

from otx.core.data.entity.base import OTXDataEntity
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module(force=True)
class LoadImageFromFile(MMCVLoadImageFromFile):
    """Class to override MMCV LoadImageFromFile."""

    def transform(self, entity: OTXDataEntity) -> dict | None:
        """Transform OTXDataEntity to MMCV data entity format."""
        img: np.ndarray = entity.image

        if self.to_float32:
            img = img.astype(np.float32)

        results = {}
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]

        results["__otx__"] = entity

        return results


class MMCVTransformLib:
    """Helper to support MMCV transforms in OTX."""

    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMCV."""
        return TRANSFORMS

    @classmethod
    def _check_mandatory_transforms(
        cls,
        transforms: list[Callable],
        mandatory_transforms: set,
    ) -> None:
        for transform in transforms:
            t_transform = type(transform)
            mandatory_transforms.discard(t_transform)

        if len(mandatory_transforms) != 0:
            msg = f"{mandatory_transforms} should be included"
            raise RuntimeError(msg)

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        """Generate MMCV transforms from the configuration."""
        transforms = [cls.get_builder().build(convert_conf_to_mmconfig_dict(cfg)) for cfg in config.transforms]

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={LoadImageFromFile},
        )

        return transforms
