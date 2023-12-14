# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMAction data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from mmaction.datasets.transforms import PackActionInputs as MMPackActionInputs
from mmaction.datasets.transforms import SampleFrames as MMSampleFrames
from mmaction.registry import TRANSFORMS
from torchvision import tv_tensors

from otx.core.data.entity.action import ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.utils.config import convert_conf_to_mmconfig_dict

if TYPE_CHECKING:
    import numpy as np
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module(force=True)
class LoadVideo:
    """Class to convert OTXDataEntity to dict for MMAction framework."""

    def __call__(self, entity: ActionClsDataEntity) -> dict:
        """Transform ActionClsDataEntity to MMAction data dictionary format."""
        video: list[np.ndarray] = entity.image

        results: dict[str, Any] = {}
        results["start_index"] = 0
        results["total_frames"] = len(video)
        results["modality"] = "RGB"
        results["imgs"] = video
        results["img_shape"] = video[0].shape[:2]
        results["ori_shape"] = video[0].shape[:2]
        results["__otx__"] = entity

        return results


@TRANSFORMS.register_module(force=True)
class SampleFrames(MMSampleFrames):
    """Class to override SampleFrames.

    MMAction's SampleFrames just sample frame indices for training.
    Actual frame sampling is done by decode pipeline.
    However, OTX already has decoded data, so here, actual sampling frame will be conducted.
    """

    def transform(self, results: dict) -> dict:
        """Transform function."""
        super().transform(results)
        imgs: list[np.ndarray] = [results["imgs"][idx] for idx in results["frame_inds"]]
        results["imgs"] = imgs

        return results


@TRANSFORMS.register_module(force=True)
class PackActionInputs(MMPackActionInputs):
    """Class to override PackActionInputs.

    Transfrom output dictionary from MMAction to ActionClsDataEntity.
    """

    def transform(self, results: dict) -> ActionClsDataEntity:
        """Transform function."""
        transformed = super().transform(results)
        video = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        img_shape = data_samples.img_shape
        ori_shape = data_samples.ori_shape
        pad_shape = data_samples.metainfo.get("pad_shape", img_shape)
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))

        labels = results["__otx__"].labels

        return ActionClsDataEntity(
            image=video,
            img_info=ImageInfo(
                img_idx=0,
                img_shape=img_shape,
                ori_shape=ori_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
            ),
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
        transforms = [
            cls.get_builder().build(convert_conf_to_mmconfig_dict(cfg))
            for cfg in config.transforms
        ]

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={LoadVideo},
        )

        return transforms
