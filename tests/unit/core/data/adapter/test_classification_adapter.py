"""Unit-Test case for otx.core.data.adapter.classification_dataset_adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
from typing import Optional

from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model_template import TaskType
from otx.api.entities.subset import Subset
from otx.core.data.adapter.classification_dataset_adapter import (
    ClassificationDatasetAdapter,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)


class TestOTXClassificationDatasetAdapter:
    def setup_method(self):
        self.root_path = os.getcwd()
        task = "classification"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        self.val_data_roots: str = os.path.join(self.root_path, data_root_dict["val"])
        self.test_data_roots: str = os.path.join(self.root_path, data_root_dict["test"])
        self.unlabeled_data_roots: Optional[str] = None
        if "unlabeled" in data_root_dict:
            self.unlabeled_data_roots = os.path.join(self.root_path, data_root_dict["unlabeled"])

        self.train_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
            unlabeled_data_roots=self.unlabeled_data_roots,
        )

        self.test_dataset_adapter = ClassificationDatasetAdapter(
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

    @e2e_pytest_unit
    def test_get_label_schema(self):
        _ = self.train_dataset_adapter.get_otx_dataset()
        assert isinstance(self.train_dataset_adapter.get_label_schema(), LabelSchemaEntity)

        _ = self.test_dataset_adapter.get_otx_dataset()
        assert isinstance(self.test_dataset_adapter.get_label_schema(), LabelSchemaEntity)

    @e2e_pytest_unit
    def test_multilabel(self):
        train_data_roots = os.path.join(self.root_path, "tests/assets/datumaro_multilabel")
        val_data_roots = os.path.join(self.root_path, "tests/assets/datumaro_multilabel")
        test_data_roots = os.path.join(self.root_path, "tests/assets/datumaro_multilabel")

        multilabel_train_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=train_data_roots,
            val_data_roots=val_data_roots,
        )

        assert Subset.TRAINING in multilabel_train_dataset_adapter.dataset
        assert Subset.VALIDATION in multilabel_train_dataset_adapter.dataset

        assert isinstance(multilabel_train_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(multilabel_train_dataset_adapter.get_label_schema(), LabelSchemaEntity)

        multilabel_test_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type, test_data_roots=test_data_roots
        )

        assert Subset.TESTING in multilabel_test_dataset_adapter.dataset
        assert isinstance(multilabel_test_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(multilabel_test_dataset_adapter.get_label_schema(), LabelSchemaEntity)

    @e2e_pytest_unit
    def test_hierarchical_label(self):
        train_data_roots = os.path.join(self.root_path, "tests/assets/datumaro_h-label")
        val_data_roots = os.path.join(self.root_path, "tests/assets/datumaro_h-label")
        test_data_roots = os.path.join(self.root_path, "tests/assets/datumaro_h-label")

        hlabel_train_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=train_data_roots,
            val_data_roots=val_data_roots,
        )

        assert Subset.TRAINING in hlabel_train_dataset_adapter.dataset
        assert Subset.VALIDATION in hlabel_train_dataset_adapter.dataset

        assert isinstance(hlabel_train_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(hlabel_train_dataset_adapter.get_label_schema(), LabelSchemaEntity)

        hlabel_test_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type, test_data_roots=test_data_roots
        )

        assert Subset.TESTING in hlabel_test_dataset_adapter.dataset
        assert isinstance(hlabel_test_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(hlabel_test_dataset_adapter.get_label_schema(), LabelSchemaEntity)
