# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMDET data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from mmdet.datasets.transforms import (
    LoadAnnotations as MMDetLoadAnnotations,
)
from mmdet.datasets.transforms import (
    PackDetInputs as MMDetPackDetInputs,
)
from mmdet.registry import TRANSFORMS
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity

from .mmcv import MMCVTransformLib

if TYPE_CHECKING:
    from mmengine.registry import Registry

    from otx.core.config.data import SubsetConfig


@TRANSFORMS.register_module(force=True)
class LoadAnnotations(MMDetLoadAnnotations):
    """Class to override MMDet LoadAnnotations."""

    def transform(self, results: dict) -> dict:
        """Transform OTXDataEntity to MMDet annotation data entity format."""
        if (otx_data_entity := results.get("__otx__")) is None:
            msg = "__otx__ key should be passed from the previous pipeline (LoadImageFromFile)"
            raise RuntimeError(msg)

        if self.with_bbox and isinstance(otx_data_entity, DetDataEntity):
            gt_bboxes = otx_data_entity.bboxes.numpy()
            results["gt_bboxes"] = gt_bboxes
        if self.with_label and isinstance(otx_data_entity, DetDataEntity):
            gt_bboxes_labels = otx_data_entity.labels.numpy()
            results["gt_bboxes_labels"] = gt_bboxes_labels - 1
            results["gt_ignore_flags"] = np.zeros_like(gt_bboxes_labels, dtype=np.bool_)

        return results


@TRANSFORMS.register_module(force=True)
class PackDetInputs(MMDetPackDetInputs):
    """Class to override PackDetInputs LoadAnnotations."""

    def transform(self, results: dict) -> DetDataEntity:
        """Pack MMDet data entity into DetDataEntity."""
        transformed = super().transform(results)

        image = tv_tensors.Image(transformed.get("inputs"))
        data_samples = transformed["data_samples"]

        img_shape = data_samples.img_shape
        ori_shape = data_samples.ori_shape
        pad_shape = data_samples.metainfo.get("pad_shape", img_shape)
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))

        bboxes = tv_tensors.BoundingBoxes(
            data_samples.gt_instances.bboxes.float(),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=img_shape,
        )
        labels = data_samples.gt_instances.labels

        return DetDataEntity(
            image=image,
            img_info=ImageInfo(
                img_idx=0,
                img_shape=img_shape,
                ori_shape=ori_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
            ),
            bboxes=bboxes,
            labels=labels,
        )


class MMDetTransformLib(MMCVTransformLib):
    """Helper to support MMDET transforms in OTX."""

    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMDet."""
        return TRANSFORMS

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        """Generate MMDET transforms from the configuration."""
        transforms = super().generate(config)

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={LoadAnnotations, PackDetInputs},
        )

        return transforms
