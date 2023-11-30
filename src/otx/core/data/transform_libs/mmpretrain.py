# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMPretrain data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from mmpretrain.datasets.transforms import (
    PackInputs as MMPretrainPackInputs,
)
from mmpretrain.registry import TRANSFORMS
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.classification import MulticlassClsDataEntity

from .mmcv import MMCVTransformLib

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig

@TRANSFORMS.register_module(force=True)
class PackInputs(MMPretrainPackInputs):
    """Class to override PackInputs."""

    def transform(self, results: dict) -> MulticlassClsDataEntity:
        """Pack MMPretrain data entity into MulticlassClsDataEntity."""
        transformed = super().transform(results)

        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        img_shape = data_samples.img_shape
        ori_shape = data_samples.ori_shape
        pad_shape = data_samples.metainfo.get("pad_shape", img_shape)
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))

        labels = results["__otx__"].labels

        return MulticlassClsDataEntity(
            image=image,
            img_info=ImageInfo(
                img_idx=0,
                img_shape=img_shape,
                ori_shape=ori_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
            ),
            labels=labels,
        )


class MMPretrainTransformLib(MMCVTransformLib):
    """Helper to support MMPretrain transforms in OTX."""

    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMPretrain."""
        return TRANSFORMS

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        """Generate MMPretrain transforms from the configuration."""
        transforms = super().generate(config)

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={PackInputs},
        )

        return transforms
