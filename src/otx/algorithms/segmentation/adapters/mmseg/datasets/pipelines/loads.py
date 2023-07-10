"""Collection of load pipelines for segmentation task."""

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
from typing import Any, Dict

from mmseg.datasets.builder import PIPELINES

import otx.core.data.pipelines.load_image_from_otx_dataset as load_image_base
from otx.algorithms.segmentation.adapters.mmseg.datasets.dataset import (
    get_annotation_mmseg_format,
)


# pylint: disable=too-many-instance-attributes, too-many-arguments
@PIPELINES.register_module()
class LoadImageFromOTXDataset(load_image_base.LoadImageFromOTXDataset):
    """Pipeline element that loads an image from a OTX Dataset on the fly."""

    def __init__(self, to_float32: bool = False, use_otx_adapter: bool = True):
        self.use_otx_adapter = use_otx_adapter
        super().__init__(to_float32)


@PIPELINES.register_module()
class LoadAnnotationFromOTXDataset:
    """Pipeline element that loads an annotation from a OTX Dataset on the fly.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the annotation
        results['ann_info']['label_list']: list of all labels in the project

    """

    def __init__(self, use_otx_adapter=True):
        self.use_otx_adapter = use_otx_adapter

    def __call__(self, results: Dict[str, Any]):
        """Callback function of LoadAnnotationFromOTXDataset."""
        dataset_item = results["dataset_item"]
        labels = results["ann_info"]["labels"]

        ann_info = get_annotation_mmseg_format(dataset_item, labels, self.use_otx_adapter)

        results["gt_semantic_seg"] = ann_info["gt_semantic_seg"]
        results["seg_fields"].append("gt_semantic_seg")

        return results
