"""Unit-Test case for otx.core.data.adapter.classification_dataset_adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path
from typing import Dict, TYPE_CHECKING, Optional

from otx.v2.adapters.datumaro.adapter.classification_dataset_adapter import (
    ClassificationDatasetAdapter,
    SelfSLClassificationDatasetAdapter,
)
from otx.v2.api.entities.label_schema import LabelSchemaEntity
from otx.v2.api.entities.subset import Subset

from tests.v2.unit.adapters.datumaro.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)

if TYPE_CHECKING:
    from otx.v2.api.entities.task_type import TaskType


class TestOTXClassificationDatasetAdapter:
    def setup_method(self) -> None:
        self.root_path = Path.cwd()
        task = "classification"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = str(self.root_path / data_root_dict["train"])
        self.val_data_roots: str = str(self.root_path / data_root_dict["val"])
        self.test_data_roots: str = str(self.root_path / data_root_dict["test"])
        self.unlabeled_data_roots: Optional[str] = None
        if "unlabeled" in data_root_dict:
            self.unlabeled_data_roots = str(self.root_path / data_root_dict["unlabeled"])

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

    def test_init(self) -> None:
        assert Subset.TRAINING in self.train_dataset_adapter.dataset
        assert Subset.VALIDATION in self.train_dataset_adapter.dataset
        if self.unlabeled_data_roots is not None:
            assert Subset.UNLABELED in self.train_dataset_adapter.dataset

        assert Subset.TESTING in self.test_dataset_adapter.dataset

    def test_get_otx_dataset(self) -> None:
        assert isinstance(self.train_dataset_adapter.get_otx_dataset(), Dict)
        assert isinstance(self.test_dataset_adapter.get_otx_dataset(), Dict)

    def test_get_label_schema(self) -> None:
        _ = self.train_dataset_adapter.get_otx_dataset()
        assert isinstance(self.train_dataset_adapter.get_label_schema(), LabelSchemaEntity)

        _ = self.test_dataset_adapter.get_otx_dataset()
        assert isinstance(self.test_dataset_adapter.get_label_schema(), LabelSchemaEntity)

    def test_multilabel(self) -> None:
        train_data_roots = str(self.root_path / "tests/assets/datumaro_multilabel")
        val_data_roots = str(self.root_path / "tests/assets/datumaro_multilabel")
        test_data_roots = str(self.root_path / "tests/assets/datumaro_multilabel")

        multilabel_train_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=train_data_roots,
            val_data_roots=val_data_roots,
        )

        assert Subset.TRAINING in multilabel_train_dataset_adapter.dataset
        assert Subset.VALIDATION in multilabel_train_dataset_adapter.dataset

        assert isinstance(multilabel_train_dataset_adapter.get_otx_dataset(), dict)
        assert isinstance(multilabel_train_dataset_adapter.get_label_schema(), LabelSchemaEntity)

        multilabel_test_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type, test_data_roots=test_data_roots,
        )

        assert Subset.TESTING in multilabel_test_dataset_adapter.dataset
        assert isinstance(multilabel_test_dataset_adapter.get_otx_dataset(), dict)
        assert isinstance(multilabel_test_dataset_adapter.get_label_schema(), LabelSchemaEntity)

    def test_hierarchical_label(self) -> None:
        train_data_roots = str(self.root_path / "tests/assets/datumaro_h-label")
        val_data_roots = str(self.root_path / "tests/assets/datumaro_h-label")
        test_data_roots = str(self.root_path / "tests/assets/datumaro_h-label")

        hlabel_train_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=train_data_roots,
            val_data_roots=val_data_roots,
        )

        assert Subset.TRAINING in hlabel_train_dataset_adapter.dataset
        assert Subset.VALIDATION in hlabel_train_dataset_adapter.dataset

        assert isinstance(hlabel_train_dataset_adapter.get_otx_dataset(), dict)
        assert isinstance(hlabel_train_dataset_adapter.get_label_schema(), LabelSchemaEntity)

        label_tree = hlabel_train_dataset_adapter.get_label_schema().label_tree
        assert label_tree.num_labels == len(hlabel_train_dataset_adapter.category_items)
        for label in label_tree.get_labels_in_topological_order():
            parent = label_tree.get_parent(label)
            parent_name = "" if parent is None else parent.name
            assert next(i.parent for i in hlabel_train_dataset_adapter.category_items if i.name == label.name) == parent_name

        hlabel_test_dataset_adapter = ClassificationDatasetAdapter(
            task_type=self.task_type, test_data_roots=test_data_roots,
        )

        assert Subset.TESTING in hlabel_test_dataset_adapter.dataset
        assert isinstance(hlabel_test_dataset_adapter.get_otx_dataset(), dict)
        assert isinstance(hlabel_test_dataset_adapter.get_label_schema(), LabelSchemaEntity)

        label_tree = hlabel_test_dataset_adapter.get_label_schema().label_tree
        assert label_tree.num_labels == len(hlabel_test_dataset_adapter.category_items)
        for label in label_tree.get_labels_in_topological_order():
            parent = label_tree.get_parent(label)
            parent_name = "" if parent is None else parent.name
            assert next(i.parent for i in hlabel_test_dataset_adapter.category_items if i.name == label.name) == parent_name


class TestSelfSLClassificationDatasetAdapter:
    def setup_method(self) -> None:
        self.root_path = Path.cwd()
        task = "classification"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        self.data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = str(self.root_path / self.data_root_dict["train"])
        self.train_data_roots_images: str = str(self.root_path / self.data_root_dict["train"] / "0")

        self.train_dataset_adapter_imagenet = SelfSLClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
        )

        self.train_dataset_adapter_images_only = SelfSLClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots_images,
        )

    def test_get_otx_dataset(self) -> None:
        dataset_imagenet = self.train_dataset_adapter_imagenet.get_otx_dataset()
        assert isinstance(dataset_imagenet, Dict)
        assert len(self.train_dataset_adapter_imagenet.get_label_schema().get_labels(False)) == 2
        dataset_only_images = self.train_dataset_adapter_images_only.get_otx_dataset()
        assert isinstance(dataset_only_images, Dict)
        lables = self.train_dataset_adapter_images_only.get_label_schema().get_labels(False)
        assert len(lables) == 1
        assert lables[0].name == "fake_label"
