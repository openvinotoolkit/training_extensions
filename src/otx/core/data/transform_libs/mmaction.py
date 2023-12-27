# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMAction data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from mmaction.datasets.transforms import PackActionInputs as MMPackActionInputs
from mmaction.registry import TRANSFORMS
from torchvision import tv_tensors

from otx.core.data.entity.action import ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module(force=True)
class LoadVideo:
    """Class to convert OTXDataEntity to dict for MMAction framework."""

    def __call__(self, entity: ActionClsDataEntity) -> dict:
        """Transform ActionClsDataEntity to MMAction data dictionary format."""
        results: dict[str, Any] = {}
        results["filename"] = entity.video.path
        results["start_index"] = 0
        results["modality"] = "RGB"
        results["__otx__"] = entity

        return results


@TRANSFORMS.register_module(force=True)
class PackActionInputs(MMPackActionInputs):
    """Class to override PackActionInputs.

    Transfrom output dictionary from MMAction to ActionClsDataEntity.
    """

    def transform(self, results: dict) -> ActionClsDataEntity:
        """Transform function."""
        transformed = super().transform(results)
        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        ori_shape = results["original_shape"]
        img_shape = data_samples.img_shape
        pad_shape = data_samples.metainfo.get("pad_shape", img_shape)
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))

        labels = results["__otx__"].labels

        data_entity = ActionClsDataEntity(
            video=results["__otx__"].video,
            image=image,
            img_info=ImageInfo(
                img_idx=0,
                img_shape=img_shape,
                ori_shape=ori_shape,
                scale_factor=scale_factor,
            ),
            labels=labels,
        )
        data_entity.img_info.pad_shape = pad_shape

        return data_entity


class MMActionTransformLib:
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
            mandatory_transforms={LoadVideo},
        )

        return transforms
