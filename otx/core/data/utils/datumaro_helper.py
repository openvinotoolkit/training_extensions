"""Datumaro Helper."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from datumaro.components.dataset import Dataset
from datumaro.components.dataset import DatasetSubset


class DatumaroHelper:
    @staticmethod
    def get_train_dataset(dataset:Dataset) -> DatasetSubset:
        """Returns train dataset."""
        for k, v in dataset.subsets().items():
            if "train" in k or "default" in k:
                return v
    @staticmethod
    def get_val_dataset(dataset:Dataset) -> DatasetSubset:
        """Returns validation dataset."""
        for k, v in dataset.subsets().items():
            if "val" in k or "default" in k:
                return v