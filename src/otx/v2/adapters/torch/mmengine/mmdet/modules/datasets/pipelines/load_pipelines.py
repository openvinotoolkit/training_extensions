"""Collection Pipeline for detection task."""
# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
from mmdet.registry import TRANSFORMS
from mmdet.structures.mask.structures import PolygonMasks

import otx.v2.adapters.torch.mmengine.modules.pipelines.transforms.pipelines as load_image_base
from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.label import Domain, LabelEntity
from otx.v2.api.entities.utils.shape_factory import ShapeFactory


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
        dataset_item: DatasetItemEntity,
        labels: list[LabelEntity],
        domain: Domain,
        min_size: int = -1,
    ) -> dict:
        """Function to convert a OTX annotation to mmdetection format.

        This is used both in the OTXDataset class defined in
        this file as in the custom pipeline element 'LoadAnnotationFromOTXDataset'

        Args:
            dataset_item: DatasetItem for which to get annotations
            labels: List of labels that are used in the task
            domain: Domain of dataset item entity; Detection, Instance Segmentation, Rotated Detection
            min_size: Minimum bbox or mask size for positive annotation
        Return
            dict: annotation information dict in mmdet format
        """
        width, height = dataset_item.width, dataset_item.height

        # load annotations for item
        gt_bboxes = []
        gt_labels = []
        gt_polygons = []
        gt_ann_ids = []

        label_idx = {label.id: i for i, label in enumerate(labels)}

        for annotation in dataset_item.get_annotations(labels=labels, include_empty=False, preserve_id=True):
            box = ShapeFactory.shape_as_rectangle(annotation.shape)

            if min(box.width * width, box.height * height) < min_size:
                continue

            class_indices = [
                label_idx[label.id] for label in annotation.get_labels(include_empty=False) if label.domain == domain
            ]

            n = len(class_indices)
            gt_bboxes.extend([[box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height] for _ in range(n)])
            if domain != Domain.DETECTION:
                polygon = ShapeFactory.shape_as_polygon(annotation.shape)
                polygon = np.array([p for point in polygon.points for p in [point.x * width, point.y * height]])
                gt_polygons.extend([[polygon] for _ in range(n)])
            gt_labels.extend(class_indices)
            item_id = getattr(dataset_item, "id_", None)
            gt_ann_ids.append((item_id, annotation.id_))

        if len(gt_bboxes) > 0:
            ann_info = {
                "bboxes": np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                "labels": np.array(gt_labels, dtype=int),
                "masks": PolygonMasks(gt_polygons, height=height, width=width) if gt_polygons else [],
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
        ann_info = self._get_annotation_mmdet_format(dataset_item, label_list, self.domain, self.min_size)
        if self.with_bbox:
            results = self._load_bboxes(results, ann_info)
        if self.with_label:
            results = self._load_labels(results, ann_info)
        if self.with_mask:
            results = self._load_masks(results, ann_info)
        return results
