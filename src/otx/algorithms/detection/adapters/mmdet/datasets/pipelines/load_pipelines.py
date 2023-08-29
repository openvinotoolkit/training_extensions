"""Collection Pipeline for detection task."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Dict, Optional

from mmdet.datasets.builder import PIPELINES, build_from_cfg
from mmdet.datasets.pipelines import Resize

import otx.algorithms.common.adapters.mmcv.pipelines.load_image_from_otx_dataset as load_image_base
from otx.algorithms.detection.adapters.mmdet.datasets.dataset import (
    get_annotation_mmdet_format,
)
from otx.api.entities.label import Domain


# pylint: disable=too-many-instance-attributes, too-many-arguments
@PIPELINES.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""


@PIPELINES.register_module()
class LoadResizeDataFromOTXDataset(load_image_base.LoadResizeDataFromOTXDataset):
    """Load and resize image & annotation with cache support."""

    def _create_load_ann_op(self, cfg: Optional[Dict]) -> Optional[Any]:
        """Creates resize operation."""
        if cfg is None:
            return None
        return build_from_cfg(cfg, PIPELINES)

    def _create_resize_op(self, cfg: Optional[Dict]) -> Optional[Any]:
        """Creates resize operation."""
        if cfg is None:
            return None
        return build_from_cfg(cfg, PIPELINES)


@PIPELINES.register_module()
class ResizeTo(Resize):
    """Resize to specific size.

    This operation works if the input is not in desired shape.
    If it's already in the shape, it just returns input dict for efficiency.

    Args:
        img_scale (tuple): Images scales for resizing (w, h).
    """

    def __init__(self, **kwargs):
        super().__init__(override=True, **kwargs)  # Allow multiple calls

    def __call__(self, results: Dict[str, Any]):
        """Callback function of ResizeTo.

        Args:
            results: Inputs to be transformed.
        """
        img_shape = results.get("img_shape", (0, 0))
        img_scale = self.img_scale[0]
        if img_shape[0] == img_scale[0] and img_shape[1] == img_scale[1]:
            return results
        return super().__call__(results)


@PIPELINES.register_module()
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
    ):
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
    def _load_bboxes(results, ann_info):
        results["bbox_fields"].append("gt_bboxes")
        results["gt_bboxes"] = copy.deepcopy(ann_info["bboxes"])
        results["gt_ann_ids"] = copy.deepcopy(ann_info["ann_ids"])
        return results

    @staticmethod
    def _load_labels(results, ann_info):
        results["gt_labels"] = copy.deepcopy(ann_info["labels"])
        return results

    @staticmethod
    def _load_masks(results, ann_info):
        results["mask_fields"].append("gt_masks")
        results["gt_masks"] = copy.deepcopy(ann_info["masks"])
        return results

    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadAnnotationFromOTXDataset."""
        dataset_item = results.pop("dataset_item")  # Prevent unnecessary deepcopy
        label_list = results.pop("ann_info")["label_list"]
        ann_info = get_annotation_mmdet_format(dataset_item, label_list, self.domain, self.min_size)
        if self.with_bbox:
            results = self._load_bboxes(results, ann_info)
            if results is None or len(results["gt_bboxes"]) == 0:
                return None
        if self.with_label:
            results = self._load_labels(results, ann_info)
        if self.with_mask:
            results = self._load_masks(results, ann_info)
        return results
