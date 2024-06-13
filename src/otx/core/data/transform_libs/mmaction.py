# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMAction data transform functions."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

from mmaction.datasets.transforms import PackActionInputs as MMPackActionInputs
from mmaction.registry import TRANSFORMS
from torchvision import tv_tensors

from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import VideoInfo
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module()
class LoadVideoForClassification:
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

    Transfrom output dictionary from MMAction to ActionClsDataEntity or ActionDetDataEntity.
    """

    def transform(self, results: dict) -> ActionClsDataEntity:
        """Transform function."""
        transformed = super().transform(results)
        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        ori_shape = results["original_shape"]
        img_shape = data_samples.img_shape
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))  # (W, H) because it is from mm pipeline

        data_entity: ActionClsDataEntity = results["__otx__"]

        image_info = deepcopy(data_entity.img_info)
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.scale_factor = scale_factor[::-1]  # convert to (H, W)

        labels = data_entity.labels

        return ActionClsDataEntity(
            video=results["__otx__"].video,
            image=image,
            img_info=image_info,
            video_info=VideoInfo(),
            labels=labels,
        )


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
        return [cls.get_builder().build(convert_conf_to_mmconfig_dict(cfg)) for cfg in config.transforms]
