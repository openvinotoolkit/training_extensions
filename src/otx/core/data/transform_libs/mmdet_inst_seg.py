# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Helper to support MMDET Instance Segmentation data transform functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from datumaro import Polygon
from mmdet.datasets.transforms import (
    LoadAnnotations as MMDetLoadAnnotations,
)
from mmdet.datasets.transforms import (
    PackDetInputs as MMDetPackDetInputs,
)
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.instance_segmentation import InstanceSegDataEntity

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

        if self.with_bbox and isinstance(otx_data_entity, InstanceSegDataEntity):
            gt_bboxes = otx_data_entity.bboxes.numpy()
            results["gt_bboxes"] = gt_bboxes
        if self.with_mask and isinstance(otx_data_entity, InstanceSegDataEntity):
            height, width = results['ori_shape']
            gt_masks = self._generate_gt_masks(otx_data_entity, height, width)
            results["gt_masks"] = gt_masks
        if self.with_label and isinstance(otx_data_entity, InstanceSegDataEntity):
            gt_bboxes_labels = otx_data_entity.labels.numpy()
            results["gt_bboxes_labels"] = gt_bboxes_labels
            results["gt_ignore_flags"] = np.zeros_like(gt_bboxes_labels, dtype=np.bool_)

        return results

    def _generate_gt_masks(
            self, otx_data_entity: InstanceSegDataEntity, height: int, width: int) -> BitmapMasks | PolygonMasks:
        """Generate ground truth masks based on the given otx_data_entity.

        Args:
            otx_data_entity (OTXDataEntity): The data entity containing the masks or polygons.
            height (int): The height of the masks.
            width (int): The width of the masks.

        Returns:
            gt_masks (BitmapMasks or PolygonMasks): The generated ground truth masks.
        """
        if len(otx_data_entity.masks):
            gt_masks = BitmapMasks(otx_data_entity.masks.numpy(), height, width)
        else:
            gt_masks = PolygonMasks(
                [[np.array(polygon.points)] for polygon in otx_data_entity.polygons], height, width,
            )
        return gt_masks


@TRANSFORMS.register_module(force=True)
class PackDetInputs(MMDetPackDetInputs):
    """Class to override PackDetInputs LoadAnnotations."""

    def transform(self, results: dict) -> InstanceSegDataEntity:
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
        image_info = ImageInfo(
                        img_idx=0,
                        img_shape=img_shape,
                        ori_shape=ori_shape,
                        pad_shape=pad_shape,
                        scale_factor=scale_factor,
                    )
        if isinstance(data_samples.gt_instances.masks, BitmapMasks):
            masks = tv_tensors.Mask(data_samples.gt_instances.masks.to_ndarray(), dtype=torch.int8)
        else:
            masks = tv_tensors.Mask(torch.empty(0))

        if isinstance(data_samples.gt_instances.masks, PolygonMasks):
            polygons = [Polygon(polygon[0]) for polygon in data_samples.gt_instances.masks.masks]
        else:
            polygons = []
        return InstanceSegDataEntity(
                image=image,
                img_info=image_info,
                bboxes=bboxes,
                masks=masks,
                labels=labels,
                polygons=polygons,
        )


class MMDetInstSegTransformLib(MMCVTransformLib):
    """Helper to support MMDET instance segmentation transform functions."""

    @classmethod
    def get_builder(cls) -> Registry:
        """Transform builder obtained from MMDet."""
        return TRANSFORMS

    @classmethod
    def generate(cls, config: SubsetConfig) -> list[Callable]:
        """Generate MMDET instance segmentation transforms from the configuration."""
        transforms = super().generate(config)

        cls._check_mandatory_transforms(
            transforms,
            mandatory_transforms={LoadAnnotations, PackDetInputs},
        )

        return transforms
