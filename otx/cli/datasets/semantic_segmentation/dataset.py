"""Module contains SemanticSegmentationDataset."""

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

from otx.algorithms.common.utils.data import load_unlabeled_dataset_items
from otx.algorithms.segmentation.adapters.mmseg.utils import load_dataset_items
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.subset import Subset


class SemanticSegmentationDataset(DatasetEntity):
    """Class for working with file-system based Semantic Segmentation dataset."""

    def __init__(
        self,
        train_subset=None,
        val_subset=None,
        test_subset=None,
        ul_subset=None,
    ):

        labels_list = []
        items = []

        if train_subset is not None:
            items.extend(
                load_dataset_items(
                    ann_file_path=train_subset["ann_file"],
                    data_root_dir=train_subset["data_root"],
                    subset=Subset.TRAINING,
                    labels_list=labels_list,
                )
            )

        if val_subset is not None:
            items.extend(
                load_dataset_items(
                    ann_file_path=val_subset["ann_file"],
                    data_root_dir=val_subset["data_root"],
                    subset=Subset.VALIDATION,
                    labels_list=labels_list,
                )
            )

        if test_subset is not None:
            items.extend(
                load_dataset_items(
                    ann_file_path=test_subset["ann_file"],
                    data_root_dir=test_subset["data_root"],
                    subset=Subset.TESTING,
                    labels_list=labels_list,
                )
            )

        if ul_subset is not None:
            items.extend(
                load_unlabeled_dataset_items(
                    file_list_path=ul_subset["file_list"],
                    data_root_dir=ul_subset["data_root"],
                    subset=Subset.UNLABELED,
                )
            )
        super().__init__(items=items)
