# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMPretrain data transform functions."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Callable

from mmseg.datasets.transforms import (
    LoadAnnotations as MMSegLoadAnnotations,
)
from mmseg.datasets.transforms import (
    PackSegInputs as MMSegPackInputs,
)
from mmseg.registry import TRANSFORMS
from torchvision import tv_tensors

from otx.core.data.entity.segmentation import SegDataEntity

from .mmcv import MMCVTransformLib

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module(force=True)
class LoadAnnotations(MMSegLoadAnnotations):
    """Class to override MMSeg LoadAnnotations."""

    def transform(self, results: dict) -> dict:
        """Transform OTXDataEntity to MMSeg annotation data entity format."""
        if (otx_data_entity := results.get("__otx__")) is None:
            msg = "__otx__ key should be passed from the previous pipeline (LoadImageFromFile)"
            raise RuntimeError(msg)
        if isinstance(otx_data_entity, SegDataEntity):
            gt_masks = otx_data_entity.gt_seg_map.numpy()
            results["gt_seg_map"] = gt_masks
            # we need this to properly handle seg maps during transforms
            results["seg_fields"] = ["gt_seg_map"]

        return results


@TRANSFORMS.register_module(force=True)
class PackSegInputs(MMSegPackInputs):
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

        image_info = deepcopy(results["__otx__"].img_info)
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.scale_factor = scale_factor
        image_info.pad_shape = pad_shape

        masks = data_samples.gt_sem_seg.data

        data_entity = SegDataEntity(
            image=image,
            img_info=image_info,
            gt_seg_map=masks,
        )
        data_entity.img_info.pad_shape = pad_shape
        return data_entity


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
            mandatory_transforms={LoadAnnotations, PackSegInputs},
        )

        return transforms
