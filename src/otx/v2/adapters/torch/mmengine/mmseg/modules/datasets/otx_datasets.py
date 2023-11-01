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

from typing import List, Optional

import numpy as np
from mmengine.registry import build_from_cfg
from mmseg.datasets import BaseCDDataset
from mmseg.registry import DATASETS, TRANSFORMS

from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label import LabelEntity
from otx.api.utils.segmentation_utils import mask_from_dataset_item


# pylint: disable=invalid-name, too-many-locals, too-many-instance-attributes, super-init-not-called
def get_annotation_mmseg_format(
    dataset_item: DatasetItemEntity,
    labels: List[LabelEntity],
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
    ann_info = dict(gt_semantic_seg=gt_seg_map)

    return ann_info


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
        empty_label: Optional[list] = None,
        pipeline: list = [],
        **kwargs,
    ):
        self.otx_dataset = otx_dataset
        self.empty_label = empty_label
        metainfo = {"classes": [lbs.name for lbs in labels]}
        test_mode = kwargs.get("test_mode", False)
        _pipeline = [{"type": "LoadImageFromOTXDataset"}, *pipeline]
        pipeline_modules = []
        for p in _pipeline:
            if isinstance(p, dict):
                pipeline_modules.append(build_from_cfg(p, TRANSFORMS))
            else:
                pipeline_modules.append(p)
        super().__init__(metainfo=metainfo, pipeline=pipeline_modules, test_mode=test_mode)

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            int: number of items in the dataset
        """
        return len(self.otx_dataset)

    def __getitem__(self, index: int) -> dict:
        """Get item from dataset."""
        dataset = self.otx_dataset
        item = dataset[index]
        ignored_labels = np.array([self.label_idx[lbs.id] + 1 for lbs in item.ignored_labels])

        data_info = dict(
            dataset_item=item,
            width=item.width,
            height=item.height,
            index=index,
            ann_info=dict(labels=self.labels),
            ignored_labels=ignored_labels,
        )

        return data_info

    def get_ann_info(self, idx: int):
        """This method is used for evaluation of predictions.

        The CustomDataset class implements a method
        CustomDataset.evaluate, which uses the class method get_ann_info to retrieve annotations.

        :param idx: index of the dataset item for which to get the annotations
        :return ann_info: dict that contains the coordinates of the bboxes and their corresponding labels
        """
        dataset_item = self.otx_dataset[idx]
        ann_info = get_annotation_mmseg_format(dataset_item, self.project_labels, self.use_otx_adapter)

        return ann_info
