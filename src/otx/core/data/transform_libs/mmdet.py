# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Helper to support MMDET data transform functions."""

from __future__ import annotations

import logging as log
from copy import deepcopy
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from datumaro import Polygon
from mmdet.datasets.transforms import LoadAnnotations as MMDetLoadAnnotations
from mmdet.datasets.transforms import PackDetInputs as MMDetPackDetInputs
from mmdet.registry import TRANSFORMS as MMDET_TRANSFORMS
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmengine.registry import Registry
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.instance_segmentation import InstanceSegDataEntity
from otx.core.data.entity.visual_prompting import VisualPromptingDataEntity

from .mmcv import MMCVTransformLib

if TYPE_CHECKING:
    from mmdet.structures.det_data_sample import DetDataSample

    from otx.core.config.data import SubsetConfig

TRANSFORMS = Registry(  # to make mmdeploy use mmdet pipeline module
    "transform",
    scope="otx",
    parent=MMDET_TRANSFORMS,
    locations=["otx.core.data.transform_libs.mmdet"],
)


@TRANSFORMS.register_module(force=True)
class LoadAnnotations(MMDetLoadAnnotations):
    """Class to override MMDet LoadAnnotations."""

    def __init__(self, with_point: bool = False, **kwargs):
        super().__init__(**kwargs)
        if with_point:
            # TODO(sungchul): add point prompts in mmx
            log.info("with_point for mmx is not supported yet, changed to False.")
            with_point = False
        self.with_point = with_point

    def transform(self, results: dict) -> dict:
        """Transform OTXDataEntity to MMDet annotation data entity format."""
        if (otx_data_entity := results.get("__otx__")) is None:
            msg = "__otx__ key should be passed from the previous pipeline (LoadImageFromFile)"
            raise RuntimeError(msg)

        if self.with_bbox and isinstance(
            otx_data_entity,
            (DetDataEntity, InstanceSegDataEntity, VisualPromptingDataEntity),
        ):
            gt_bboxes = otx_data_entity.bboxes.numpy()
            results["gt_bboxes"] = gt_bboxes
        if self.with_label and isinstance(
            otx_data_entity,
            (DetDataEntity, InstanceSegDataEntity, VisualPromptingDataEntity),
        ):
            gt_bboxes_labels = otx_data_entity.labels.numpy()  # type: ignore[union-attr]
            results["gt_bboxes_labels"] = gt_bboxes_labels
            results["gt_ignore_flags"] = np.zeros_like(gt_bboxes_labels, dtype=np.bool_)
        if self.with_mask and isinstance(otx_data_entity, (InstanceSegDataEntity, VisualPromptingDataEntity)):
            height, width = results["ori_shape"]
            gt_masks = self._generate_gt_masks(otx_data_entity, height, width)
            results["gt_masks"] = gt_masks
        if self.with_point and isinstance(otx_data_entity, (VisualPromptingDataEntity)):
            # TODO(sungchul): add point prompts in mmx
            # gt_points = otx_data_entity.points.numpy()
            # results["gt_points"] = gt_points
            pass
        return results

    def _generate_gt_masks(
        self,
        otx_data_entity: InstanceSegDataEntity | VisualPromptingDataEntity,
        height: int,
        width: int,
    ) -> BitmapMasks | PolygonMasks:
        """Generate ground truth masks based on the given otx_data_entity.

        Args:
            otx_data_entity (OTXDataEntity): The data entity containing the masks or polygons.
            height (int): The height of the masks.
            width (int): The width of the masks.

        Returns:
            gt_masks (BitmapMasks or PolygonMasks): The generated ground truth masks.
        """
        if len(otx_data_entity.masks):
            return BitmapMasks(otx_data_entity.masks.numpy(), height, width)

        return PolygonMasks(
            [[np.array(polygon.points)] for polygon in otx_data_entity.polygons],
            height,
            width,
        )


@TRANSFORMS.register_module(force=True)
class PackDetInputs(MMDetPackDetInputs):
    """Class to override PackDetInputs LoadAnnotations."""

    def transform(self, results: dict) -> DetDataEntity | InstanceSegDataEntity | VisualPromptingDataEntity:
        """Pack MMDet data entity into DetDataEntity, InstanceSegDataEntity, or VisualPromptingDataEntity."""
        otx_data_entity = results["__otx__"]

        if isinstance(otx_data_entity, DetDataEntity):
            return self.pack_det_inputs(results)
        if isinstance(otx_data_entity, InstanceSegDataEntity):
            return self.pack_inst_inputs(results)
        if isinstance(otx_data_entity, VisualPromptingDataEntity):
            return self.pack_visprompt_inputs(results)
        msg = "Unsupported data entity type"
        raise TypeError(msg)

    def pack_det_inputs(self, results: dict) -> DetDataEntity:
        """Pack MMDet data entity into DetDataEntity."""
        transformed = super().transform(results)
        data_samples = transformed["data_samples"]
        image_info = self.create_image_info(src_image_info=results["__otx__"].img_info, data_samples=data_samples)

        bboxes = self.convert_bboxes(data_samples.gt_instances.bboxes, image_info.img_shape)
        labels = data_samples.gt_instances.labels

        return DetDataEntity(
            image=tv_tensors.Image(transformed.get("inputs")),
            img_info=image_info,
            bboxes=bboxes,
            labels=labels,
        )

    def pack_inst_inputs(self, results: dict) -> InstanceSegDataEntity:
        """Pack MMDet data entity into InstanceSegDataEntity."""
        transformed = super().transform(results)
        data_samples = transformed["data_samples"]
        image_info = self.create_image_info(src_image_info=results["__otx__"].img_info, data_samples=data_samples)

        bboxes = self.convert_bboxes(data_samples.gt_instances.bboxes, image_info.img_shape)
        labels = data_samples.gt_instances.labels

        masks, polygons = self.convert_masks_and_polygons(data_samples.gt_instances.masks)

        return InstanceSegDataEntity(
            image=tv_tensors.Image(transformed.get("inputs")),
            img_info=image_info,
            bboxes=bboxes,
            masks=masks,
            labels=labels,
            polygons=polygons,
        )

    def pack_visprompt_inputs(self, results: dict) -> VisualPromptingDataEntity:
        """Pack MMDet data entity into VisualPromptingDataEntity."""
        transformed = super().transform(results)
        data_samples = transformed["data_samples"]
        image_info = self.create_image_info(src_image_info=results["__otx__"].img_info, data_samples=data_samples)

        bboxes = self.convert_bboxes(data_samples.gt_instances.bboxes, image_info.img_shape)
        labels = data_samples.gt_instances.labels

        return VisualPromptingDataEntity(
            image=tv_tensors.Image(transformed.get("inputs")),
            img_info=image_info,
            bboxes=bboxes,
            points=None,  # type: ignore[arg-type]
            masks=None,
            labels=labels,
            polygons=None,  # type: ignore[arg-type]
        )

    def create_image_info(
        self,
        src_image_info: ImageInfo,
        data_samples: DetDataSample,
    ) -> ImageInfo:
        """Create ImageInfo instance from data_samples."""
        # Some MM* transforms return (H, W, C), not (H, W)
        img_shape = data_samples.img_shape if len(data_samples.img_shape) == 2 else data_samples.img_shape[:2]
        ori_shape = data_samples.ori_shape if len(data_samples.ori_shape) == 2 else data_samples.ori_shape[:2]
        scale_factor = data_samples.metainfo.get("scale_factor", (1.0, 1.0))  # (W, H) because it is from mm pipeline

        image_info = deepcopy(src_image_info)
        image_info.img_shape = img_shape
        image_info.ori_shape = ori_shape
        image_info.scale_factor = scale_factor[::-1]  # convert to (H, W)

        return image_info

    def convert_bboxes(self, original_bboxes: torch.Tensor, img_shape: tuple[int, int]) -> tv_tensors.BoundingBoxes:
        """Convert bounding boxes to tv_tensors.BoundingBoxes format."""
        return tv_tensors.BoundingBoxes(
            original_bboxes.float(),
            format=tv_tensors.BoundingBoxFormat.XYXY,
            canvas_size=img_shape,
        )

    def convert_masks_and_polygons(self, masks: BitmapMasks | PolygonMasks) -> tuple[tv_tensors.Mask, list[Polygon]]:
        """Convert masks and polygons to the desired format."""
        if isinstance(masks, BitmapMasks):
            masks_tensor = tv_tensors.Mask(masks.to_ndarray(), dtype=torch.int8)
        else:
            masks_tensor = tv_tensors.Mask(torch.empty(0))

        polygons = [Polygon(polygon[0]) for polygon in masks.masks] if isinstance(masks, PolygonMasks) else []

        return masks_tensor, polygons


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
