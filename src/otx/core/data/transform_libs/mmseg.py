# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMPretrain data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from mmseg.datasets.transforms import (
    PackSegInputs as MMSegPackInputs,
)

from mmseg.registry import TRANSFORMS
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import SegDataEntity

from .mmcv import MMCVTransformLib

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig

@TRANSFORMS.register_module(force=True)
class PackInputs(MMSegPackInputs):
    """Class to override PackInputs."""

    def transform(self, results: dict) -> SegDataEntity:
        """Pack MMSeg data entity into SegDataEntity."""
        transformed = super().transform(results)

        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        img_shape = data_samples.img_shape
        ori_shape = data_samples.ori_shape
        pad_shape = data_samples.metainfo.get("pad_shape", img_shape)
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))

        masks = results["__otx__"].masks

        return SegDataEntity(
            image=image,
            img_info=ImageInfo(
                img_idx=0,
                img_shape=img_shape,
                ori_shape=ori_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
            ),
            masks=masks,
        )


class MMSegTransformLib(MMCVTransformLib):
    """Helper to support MMSeg transforms in OTX."""

    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMSeg."""
        return TRANSFORMS

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        """Generate MMSeg transforms from the configuration."""
        transforms = super().generate(config)

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={PackInputs},
        )

        return transforms
