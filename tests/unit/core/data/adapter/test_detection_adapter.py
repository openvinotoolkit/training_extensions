"""Unit-Test case for otx.core.data.adapter.detection_dataset_adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from typing import Optional

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.core.data.adapter.detection_dataset_adapter import DetectionDatasetAdapter
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)


class TestOTXDetectionDatasetAdapter:
    def setup_method(self):
        self.root_path = os.getcwd()
        task = "detection"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        self.val_data_roots: str = os.path.join(self.root_path, data_root_dict["val"])
        self.test_data_roots: str = os.path.join(self.root_path, data_root_dict["test"])
        self.unlabeled_data_roots: Optional[str] = None
        if "unlabeled" in data_root_dict:
            self.unlabeled_data_roots = os.path.join(self.root_path, data_root_dict["unlabeled"])

        self.train_dataset_adapter = DetectionDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
            unlabeled_data_roots=self.unlabeled_data_roots,
        )

        self.test_dataset_adapter = DetectionDatasetAdapter(
            task_type=self.task_type,
            test_data_roots=self.test_data_roots,
        )

    @e2e_pytest_unit
    def test_init(self):
        assert Subset.TRAINING in self.train_dataset_adapter.dataset
        assert Subset.VALIDATION in self.train_dataset_adapter.dataset
        assert Subset.TESTING in self.test_dataset_adapter.dataset
        if self.unlabeled_data_roots is not None:
            assert Subset.UNLABELED in self.train_dataset_adapter.dataset

    @e2e_pytest_unit
    def test_get_otx_dataset(self):
        assert isinstance(self.train_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(self.test_dataset_adapter.get_otx_dataset(), DatasetEntity)

    @e2e_pytest_unit
    def test_instance_segmentation(self):
        task = "instance_segmentation"

        task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        val_data_roots: str = os.path.join(self.root_path, data_root_dict["val"])
        test_data_roots: str = os.path.join(self.root_path, data_root_dict["test"])

        instance_seg_train_dataset_adapter = DetectionDatasetAdapter(
            task_type=task_type,
            train_data_roots=train_data_roots,
            val_data_roots=val_data_roots,
        )

        assert Subset.TRAINING in instance_seg_train_dataset_adapter.dataset
        assert Subset.VALIDATION in instance_seg_train_dataset_adapter.dataset

        assert isinstance(instance_seg_train_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(instance_seg_train_dataset_adapter.get_label_schema(), LabelSchemaEntity)

        instance_seg_test_dataset_adapter = DetectionDatasetAdapter(
            task_type=task_type,
            test_data_roots=test_data_roots,
        )

        assert Subset.TESTING in instance_seg_test_dataset_adapter.dataset
        assert isinstance(instance_seg_test_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(instance_seg_test_dataset_adapter.get_label_schema(), LabelSchemaEntity)
