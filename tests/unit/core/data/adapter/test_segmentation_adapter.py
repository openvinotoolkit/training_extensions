"""Unit-Test case for otx.core.data.adapter.segmentation_dataset_adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from typing import Optional

import numpy as np
import pytest

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.core.data.adapter.segmentation_dataset_adapter import (
    SegmentationDatasetAdapter,
    SelfSLSegmentationDatasetAdapter,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)


class TestOTXSegmentationDatasetAdapter:
    def setup_method(self):
        self.root_path = os.getcwd()
        task = "segmentation"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        self.val_data_roots: str = os.path.join(self.root_path, data_root_dict["val"])
        self.test_data_roots: str = os.path.join(self.root_path, data_root_dict["test"])
        self.unlabeled_data_roots: Optional[str] = None
        if "unlabeled" in data_root_dict:
            self.unlabeled_data_roots = os.path.join(self.root_path, data_root_dict["unlabeled"])

        self.train_dataset_adapter = SegmentationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
            unlabeled_data_roots=self.unlabeled_data_roots,
        )

        self.test_dataset_adapter = SegmentationDatasetAdapter(
            task_type=self.task_type,
            test_data_roots=self.test_data_roots,
        )

    @e2e_pytest_unit
    def test_init(self):
        assert Subset.TRAINING in self.train_dataset_adapter.dataset
        assert Subset.VALIDATION in self.train_dataset_adapter.dataset
        if self.unlabeled_data_roots is not None:
            assert Subset.UNLABELED in self.train_dataset_adapter.dataset

        assert Subset.TESTING in self.test_dataset_adapter.dataset

    @e2e_pytest_unit
    def test_get_otx_dataset(self):
        assert isinstance(self.train_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(self.test_dataset_adapter.get_otx_dataset(), DatasetEntity)


class TestSelfSLSegmentationDatasetAdapter:

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.root_path = os.getcwd()
        task = "segmentation"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]
        self.train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        self.dataset_adapter = SelfSLSegmentationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
        )
    
    @e2e_pytest_unit
    def test_import_dataset(self):
        """Test _import_dataset.

        - create all images
        - create some uncreated images due to unknown issues (others were already created)
        - just load created images

        TODO (sungchul): don't skip background class in get_otx_dataset
        """
        pass

    @e2e_pytest_unit
    def test_create_pseudo_masks(self, mocker):
        """Test create_pseudo_masks."""
        mocker.patch("otx.core.data.adapter.segmentation_dataset_adapter.os.makedirs")
        mocker.patch("otx.core.data.adapter.segmentation_dataset_adapter.cv2.imwrite")

        pseudo_mask = self.dataset_adapter.create_pseudo_masks(img=np.ones((2, 2)), pseudo_mask_path="")

        assert type(pseudo_mask) == np.ndarray
