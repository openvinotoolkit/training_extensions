"""
Module contains ObjectDetectionDataset
"""

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

from torchreid.integration.sc.utils import ClassificationDatasetAdapter


class ImageClassificationDataset(ClassificationDatasetAdapter):
    """Class for working with file-system based Image Classification dataset."""

    def __init__(
        self,
        train_subset=None,
        val_subset=None,
        test_subset=None,
    ):
        super().__init__(
            train_subset.get("ann_file", None) if train_subset else None,
            train_subset.get("data_root", None) if train_subset else None,
            val_subset.get("ann_file", None) if val_subset else None,
            val_subset.get("data_root", None) if val_subset else None,
            test_subset.get("ann_file", None) if test_subset else None,
            test_subset.get("data_root", None) if test_subset else None,
        )
