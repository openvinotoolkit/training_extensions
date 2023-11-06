"""Base MMDataset for Segmentation Task."""

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
from mmseg.datasets import BaseCDDataset

from otx.v2.adapters.torch.mmengine.mmseg.registry import DATASETS, TRANSFORMS
from otx.v2.api.entities.dataset_item import DatasetItemEntity
from otx.v2.api.entities.datasets import DatasetEntity
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


@DATASETS.register_module()
class OTXSegDataset(BaseCDDataset):
    """Wrapper that allows using a OTX dataset to train mmsegmentation models.

    This wrapper is not based on the filesystem,
    but instead loads the items here directly from the OTX Dataset object.

    The wrapper overwrites some methods of the CustomDataset class: prepare_train_img, prepare_test_img and prepipeline
    Naming of certain attributes might seem a bit peculiar but this is due to the conventions set in CustomDataset. For
    instance, CustomDatasets expects the dataset items to be stored in the attribute data_infos, which is why it is
    named like that and not dataset_items.

    """

    def __init__(
        self,
        otx_dataset: DatasetEntity,
        labels: list[LabelEntity],
        empty_label: list | None = None,
        pipeline: list | dict | None = None,
        **kwargs,
    ) -> None:
        """Dataset class for OTX datasets.

        Args:
            otx_dataset (DatasetEntity): The OTX dataset to use.
            labels (list[LabelEntity]): List of label entities.
            empty_label (list | None, optional): Empty label. Defaults to None.
            pipeline (list | dict | None, optional): Data processing pipeline. Defaults to None.
            **kwargs: Additional keyword arguments.

        Attributes:
            otx_dataset (DatasetEntity): The OTX dataset being used.
            empty_label (list | None): Empty label.
            labels (list[LabelEntity]): List of label entities.
            serialize_data (None): OTX has its own data caching mechanism.
            _fully_initialized (bool): Whether the dataset has been fully initialized.
        """
        self.otx_dataset = otx_dataset
        self.empty_label = empty_label
        self.labels = labels
        metainfo = {"classes": [lbs.name for lbs in labels]}
        test_mode = kwargs.get("test_mode", False)
        super().__init__(metainfo=metainfo, pipeline=pipeline, test_mode=test_mode, lazy_init=True)
        self.serialize_data = None  # OTX has its own data caching mechanism
        self._fully_initialized = True

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: number of items in the dataset
        """
        return len(self.otx_dataset)

    def prepare_data(self, idx: int) -> dict:
        """Get item from dataset."""
        dataset = self.otx_dataset
        item = dataset[idx]
        ignored_labels = np.array([self.label_idx[lbs.id] + 1 for lbs in item.ignored_labels])

        data_info = {
            "dataset_item": item,
            "width": item.width,
            "height": item.height,
            "index": idx,
            "ann_info": {"labels": self.labels},
            "ignored_labels": ignored_labels,
            "seg_fields": [],
        }
        return self.pipeline(data_info)


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
