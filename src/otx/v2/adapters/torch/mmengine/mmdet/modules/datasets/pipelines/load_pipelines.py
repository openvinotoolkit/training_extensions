"""Collection Pipeline for detection task."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
from mmdet.datasets.transforms import Resize
from mmdet.registry import TRANSFORMS

import otx.v2.adapters.torch.mmengine.modules.pipelines.transforms.pipelines as load_image_base
from otx.v2.adapters.torch.mmengine.mmdet.modules.datasets.dataset import (
    get_annotation_mmdet_format,
)
from otx.v2.api.entities.label import Domain


# pylint: disable=too-many-instance-attributes, too-many-arguments
@TRANSFORMS.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""


@TRANSFORMS.register_module()
class LoadResizeDataFromOTXDataset(load_image_base.LoadResizeDataFromOTXDataset):
    """Load and resize image & annotation with cache support."""

    def _create_load_ann_op(self, cfg: dict | None) -> Callable | None:
        """Creates resize operation."""
        if cfg is None:
            return None
        return TRANSFORMS.build(cfg)

    def _create_resize_op(self, cfg: dict | None) -> Callable | None:
        """Creates resize operation."""
        if cfg is None:
            return None
        return TRANSFORMS.build(cfg)


@TRANSFORMS.register_module()
class ResizeTo(Resize):
    """Resize to specific size.

    This operation works if the input is not in desired shape.
    If it's already in the shape, it just returns input dict for efficiency.
    """

    def __init__(self, **kwargs) -> None:
        """Initialize method.

        Args:
            kwargs: Additional kwargs for parent classes
        """
        super().__init__(override=True, **kwargs)  # Allow multiple calls

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        """Callback function of ResizeTo.

        Args:
            results: Inputs to be transformed.
        """
        img_shape = results.get("img_shape", (0, 0))
        img_scale = self.img_scale[0]
        if img_shape[0] == img_scale[0] and img_shape[1] == img_scale[1]:
            return results
        return super().__call__(results)


@TRANSFORMS.register_module()
class LoadAnnotationFromOTXDataset:
    """Pipeline element that loads an annotation from a OTX Dataset on the fly.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the annotation
        results['ann_info']['label_list']: list of all labels in the project
    """

    def __init__(
        self,
        min_size: int = -1,
        with_bbox: bool = True,
        with_label: bool = True,
        with_mask: bool = False,
        with_seg: bool = False,
        poly2mask: bool = True,
        with_text: bool = False,
        domain: str = "detection",
    ) -> None:
        """Initialize method.

        Args:
            min_size (int): Minimum size for loading data. Default = -1
            with_bbox (bool): Whether loading ground truth bounding box information or not. Default = True
            with_label (bool): Whether loading ground truth labels or not. Default = True
            with_mask (bool): Whether loading ground truth masks or not. Default = False
            with_seg (bool): Whether loading ground truth segmenation or not. Default = False
            poly2mask (bool): Whether converting polygon to mask or not. Default = True
            with_text (bool): Whether loading text information or not. Defualt = True
            domain (str): Domain of loading pipeline. detection or instance-segmentation
        """
        self._domain_dict = {
            "detection": Domain.DETECTION,
            "instance_segmentation": Domain.INSTANCE_SEGMENTATION,
            "rotated_detection": Domain.ROTATED_DETECTION,
        }
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.with_text = with_text
        self.domain = self._domain_dict[domain.lower()]
        self.min_size = min_size

    @staticmethod
    def _load_bboxes(results: dict[str, Any], ann_info: dict[str, Any]) -> dict[str, Any]:
        results["bbox_fields"].append("gt_bboxes")
        results["gt_bboxes"] = copy.deepcopy(ann_info["bboxes"])
        results["gt_ann_ids"] = copy.deepcopy(ann_info["ann_ids"])
        results["gt_ignore_flags"] = np.array([False] * len(results["gt_bboxes"]))
        return results

    @staticmethod
    def _load_labels(results: dict[str, Any], ann_info: dict[str, Any]) -> dict[str, Any]:
        results["gt_bboxes_labels"] = copy.deepcopy(ann_info["labels"])
        return results

    @staticmethod
    def _load_masks(results: dict[str, Any], ann_info: dict[str, Any]) -> dict[str, Any]:
        results["mask_fields"].append("gt_masks")
        results["gt_masks"] = copy.deepcopy(ann_info["masks"])
        return results

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        """Callback function of LoadAnnotationFromOTXDataset."""
        dataset_item = results.pop("dataset_item")  # Prevent unnecessary deepcopy
        label_list = results.pop("ann_info")["label_list"]
        ann_info = get_annotation_mmdet_format(dataset_item, label_list, self.domain, self.min_size)
        if self.with_bbox:
            results = self._load_bboxes(results, ann_info)
        if self.with_label:
            results = self._load_labels(results, ann_info)
        if self.with_mask:
            results = self._load_masks(results, ann_info)
        return results
