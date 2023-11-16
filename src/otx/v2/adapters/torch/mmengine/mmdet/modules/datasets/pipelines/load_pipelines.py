"""Collection Pipeline for detection task."""

# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from datumaro.components.annotation import Polygon
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask.structures import PolygonMasks

if TYPE_CHECKING:
    from datumaro.components.dataset_base import DatasetItem

import otx.v2.adapters.torch.mmengine.modules.pipelines.transforms.pipelines as load_image_base
from otx.v2.api.entities.label import Domain, LabelEntity


@TRANSFORMS.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""


@TRANSFORMS.register_module()
class LoadResizeDataFromOTXDataset(load_image_base.LoadResizeDataFromOTXDataset):
    """Load and resize image & annotation with cache support."""

    def _create_load_ann_op(self, cfg: dict | None = None) -> Callable | None:
        """Creates resize operation."""
        return TRANSFORMS.build(cfg) if cfg else None

    def _create_resize_op(self, cfg: dict | None = None) -> Callable | None:
        """Creates resize operation."""
        return TRANSFORMS.build(cfg) if cfg else None


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
    def _get_annotation_mmdet_format(
        dataset_item: DatasetItem,
        labels: list[LabelEntity],
        min_size: int = -1,
    ) -> dict:
        """Function to convert a OTX annotation to mmdetection format.

        This is used both in the OTXDataset class defined in
        this file as in the custom pipeline element 'LoadAnnotationFromOTXDataset'

        Args:
            dataset_item: DatasetItem for which to get annotations
            labels: List of labels that are used in the task
            min_size: Minimum bbox or mask size for positive annotation
        Return
            dict: annotation information dict in mmdet format
        """
        # load annotations for item
        gt_bboxes: list[list[float]] = []
        gt_labels: list[int] = []
        gt_polygons: list[list[np.array]] = []
        gt_ann_ids: list[tuple[str, int]] = []

        image_height, image_width = dataset_item.media.data.shape[:2]
        _labels: list[int] = [int(label.id) for label in labels]

        for annotation in dataset_item.annotations:
            if annotation.label not in _labels:
                continue

            if isinstance(annotation, Polygon):
                gt_polygons.append([np.asarray(annotation.points)])
                bbox = annotation.get_bbox()
                annotation_width = bbox[2]
                annotation_height = bbox[3]
                # [x1, y1, w, h] -> [x1, y1, x2, y2]
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            else:
                bbox = annotation.points
                annotation_width = annotation.w
                annotation_height = annotation.h

            if min(annotation_width, annotation_height) < min_size:
                continue

            gt_bboxes.append(bbox)
            gt_labels.append(annotation.label)
            gt_ann_ids.append((dataset_item.id, annotation.id))

        if len(gt_bboxes) > 0:
            ann_info = {
                "bboxes": np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                "labels": np.array(gt_labels, dtype=int),
                "masks": PolygonMasks(gt_polygons, height=image_height, width=image_width) if gt_polygons else [],
                "ann_ids": gt_ann_ids,
            }
        else:
            ann_info = {
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.array([], dtype=int),
                "masks": np.zeros((0, 1), dtype=np.float32),
                "ann_ids": [],
            }
        return ann_info

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
        ann_info = self._get_annotation_mmdet_format(dataset_item, label_list, self.min_size)
        if self.with_bbox:
            results = self._load_bboxes(results, ann_info)
        if self.with_label:
            results = self._load_labels(results, ann_info)
        if self.with_mask:
            results = self._load_masks(results, ann_info)
        return results
