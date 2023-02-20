"""Unit-Test case for otx.core.data.adapter.action_dataset_adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.core.data.adapter.action_dataset_adapter import (
    ActionClassificationDatasetAdapter,
    ActionDetectionDatasetAdapter,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)


class TestOTXActionClassificationDatasetAdapter:
    def setup_method(self):
        self.root_path = os.getcwd()
        task = "action_classification"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        self.val_data_roots: str = os.path.join(self.root_path, data_root_dict["val"])
        self.test_data_roots: str = os.path.join(self.root_path, data_root_dict["test"])

        self.train_dataset_adapter = ActionClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
        )

        self.test_dataset_adapter = ActionClassificationDatasetAdapter(
            task_type=self.task_type,
            test_data_roots=self.test_data_roots,
        )

    @e2e_pytest_unit
    def test_init(self):
        assert Subset.TRAINING in self.train_dataset_adapter.dataset
        assert Subset.VALIDATION in self.train_dataset_adapter.dataset
        assert Subset.TESTING in self.test_dataset_adapter.dataset

    @e2e_pytest_unit
    def test_get_otx_dataset(self):
        assert isinstance(self.train_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(self.test_dataset_adapter.get_otx_dataset(), DatasetEntity)


class TestOTXActionDetectionDatasetAdapter:
    def setup_method(self):
        self.root_path = os.getcwd()
        task = "action_detection"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        self.val_data_roots: str = os.path.join(self.root_path, data_root_dict["val"])
        self.test_data_roots: str = os.path.join(self.root_path, data_root_dict["test"])

        self.train_dataset_adapter = ActionDetectionDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
        )

        self.test_dataset_adapter = ActionDetectionDatasetAdapter(
            task_type=self.task_type,
            test_data_roots=self.test_data_roots,
        )

    @e2e_pytest_unit
    def test_init(self):
        assert Subset.TRAINING in self.train_dataset_adapter.dataset
        assert Subset.VALIDATION in self.train_dataset_adapter.dataset
        assert Subset.TESTING in self.test_dataset_adapter.dataset

    @e2e_pytest_unit
    def test_get_otx_dataset(self):
        assert isinstance(self.train_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(self.test_dataset_adapter.get_otx_dataset(), DatasetEntity)
