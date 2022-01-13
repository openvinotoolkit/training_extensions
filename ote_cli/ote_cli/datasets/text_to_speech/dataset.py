"""
Module contains ImageClassificationDataset
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


class EmptyTTSDataset():
    """Class for working with file-system based text to speech dataset."""

    def __init__(
        self,
        train_subset=None,
        val_subset=None,
        test_subset=None,
    ):
        self.train_ann_file = train_subset.get("ann_file", None) if train_subset else None
        self.train_data_root = train_subset.get("data_root", None) if train_subset else None
        self.val_ann_file = val_subset.get("ann_file", None) if val_subset else None
        self.val_data_root = val_subset.get("data_root", None) if val_subset else None
        self.test_ann_file = test_subset.get("ann_file", None) if test_subset else None
        self.test_root_file = test_subset.get("data_root", None) if test_subset else None
