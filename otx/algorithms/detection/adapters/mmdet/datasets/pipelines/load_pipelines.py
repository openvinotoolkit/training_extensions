"""Collection Pipeline for detection task."""
# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
import copy
from typing import Any, Dict, Optional

from mmdet.datasets.builder import PIPELINES

import otx.core.data.pipelines.load_image_from_otx_dataset as load_image_base
from otx.algorithms.detection.adapters.mmdet.datasets.dataset import (
    get_annotation_mmdet_format,
)
from otx.api.entities.label import Domain
from otx.api.utils.argument_checks import check_input_parameters_type


# pylint: disable=too-many-instance-attributes, too-many-arguments
@PIPELINES.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""


@PIPELINES.register_module()
class LoadAnnotationFromOTXDataset:
    """Pipeline element that loads an annotation from a OTX Dataset on the fly.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the annotation
        results['ann_info']['label_list']: list of all labels in the project
    """

    @check_input_parameters_type()
    def __init__(
        self,
        min_size: int,
        with_bbox: bool = True,
        with_label: bool = True,
        with_mask: bool = False,
        with_seg: bool = False,
        poly2mask: bool = True,
        with_text: bool = False,
        domain: Optional[Domain] = None,
    ):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.with_text = with_text
        self.domain = domain
        self.min_size = min_size

    @staticmethod
    def _load_bboxes(results, ann_info):
        results["bbox_fields"].append("gt_bboxes")
        results["gt_bboxes"] = copy.deepcopy(ann_info["bboxes"])
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

    @check_input_parameters_type()
    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadAnnotationFromOTXDataset."""
        dataset_item = results["dataset_item"]
        label_list = results["ann_info"]["label_list"]
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
