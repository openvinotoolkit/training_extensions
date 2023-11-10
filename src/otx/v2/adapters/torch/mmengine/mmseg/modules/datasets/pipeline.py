"""Data Pipeline for MMSegmentation Task."""

# Copyright (C) 2023 Intel Corporation
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

from __future__ import annotations

from typing import Any

import numpy as np

from otx.v2.adapters.torch.mmengine.mmseg.registry import TRANSFORMS
from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.label import LabelEntity
from otx.v2.api.entities.utils.segmentation_utils import mask_from_dataset_item


def get_annotation_mmseg_format(
    dataset_item: DatasetItemEntity,
    labels: list[LabelEntity],
    use_otx_adapter: bool = True,
) -> dict:
    """Function to convert a OTX annotation to mmsegmentation format.

    This is used both in the OTXDataset class defined in this file
    as in the custom pipeline element 'LoadAnnotationFromOTXDataset'

    :param dataset_item: DatasetItem for which to get annotations
    :param labels: List of labels in the project
    :return dict: annotation information dict in mmseg format
    """
    gt_seg_map = mask_from_dataset_item(dataset_item, labels, use_otx_adapter)
    gt_seg_map = gt_seg_map.squeeze(2).astype(np.uint8)
    return {"gt_seg_map": gt_seg_map}


@TRANSFORMS.register_module()
class LoadAnnotationFromOTXDataset:
    """Pipeline element that loads an annotation from a OTX Dataset on the fly.

    Expected entries in the 'results' dict that should be passed to this pipeline element are:
        results['dataset_item']: dataset_item from which to load the annotation
        results['ann_info']['label_list']: list of all labels in the project

    """

    def __init__(self, use_otx_adapter: bool = True) -> None:
        """Initializes the pipeline element.

        Args:
            use_otx_adapter (bool): Whether to use the OTX adapter or not.
        """
        self.use_otx_adapter = use_otx_adapter

    def __call__(self, results: dict[str, Any]) -> dict:
        """Callback function of LoadAnnotationFromOTXDataset."""
        dataset_item = results.pop("dataset_item")  # Prevent unnessary deepcopy
        labels = results["ann_info"]["labels"]

        ann_info = get_annotation_mmseg_format(dataset_item, labels, self.use_otx_adapter)

        results["gt_seg_map"] = ann_info["gt_seg_map"]
        results["seg_fields"].append("gt_seg_map")

        return results
