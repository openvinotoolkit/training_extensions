"""Unit-Test case for otx.core.data.adapter.detection_dataset_adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os

from otx.api.entities.annotation import NullAnnotationSceneEntity
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

    @e2e_pytest_unit
    def test_detection(self):
        task = "detection"

        task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        val_data_roots: str = os.path.join(self.root_path, data_root_dict["val"])
        test_data_roots: str = os.path.join(self.root_path, data_root_dict["test"])

        det_train_dataset_adapter = DetectionDatasetAdapter(
            task_type=task_type,
            train_data_roots=train_data_roots,
            val_data_roots=val_data_roots,
        )

        assert Subset.TRAINING in det_train_dataset_adapter.dataset
        assert Subset.VALIDATION in det_train_dataset_adapter.dataset

        det_train_dataset = det_train_dataset_adapter.get_otx_dataset()
        det_train_label_schema = det_train_dataset_adapter.get_label_schema()
        assert isinstance(det_train_dataset, DatasetEntity)
        assert isinstance(det_train_label_schema, LabelSchemaEntity)

        # In the test data, there is a empty_label image.
        # So, has_empty_label should be True
        has_empty_label = False
        for train_data in det_train_dataset:
            if isinstance(train_data.annotation_scene, NullAnnotationSceneEntity):
                has_empty_label = True
        assert has_empty_label is True

        det_test_dataset_adapter = DetectionDatasetAdapter(
            task_type=task_type,
            test_data_roots=test_data_roots,
        )

        assert Subset.TESTING in det_test_dataset_adapter.dataset
        assert isinstance(det_test_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(det_test_dataset_adapter.get_label_schema(), LabelSchemaEntity)

    @e2e_pytest_unit
    def test_get_subset_data(self):
        class MockDatumDataset:
            def __init__(self, subsets):
                self._subsets = subsets
                self.eager = None

            def subsets(self):
                return self._subsets

        class MockDataset:
            def __init__(self, string):
                self.string = string

            def as_dataset(self):
                return self.string

        task = "detection"

        task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"])
        val_data_roots: str = os.path.join(self.root_path, data_root_dict["val"])

        det_train_dataset_adapter = DetectionDatasetAdapter(
            task_type=task_type,
            train_data_roots=train_data_roots,
            val_data_roots=val_data_roots,
        )

        dataset = MockDatumDataset(
            {
                "train": MockDataset("train"),
                "val": MockDataset("val"),
                "test": MockDataset("test"),
            }
        )
        assert det_train_dataset_adapter._get_subset_data("val", dataset) == "val"

        dataset = MockDatumDataset(
            {
                "train": MockDataset("train"),
                "validation": MockDataset("validation"),
                "test": MockDataset("test"),
            }
        )
        assert det_train_dataset_adapter._get_subset_data("val", dataset) == "validation"

        dataset = MockDatumDataset(
            {
                "train": MockDataset("train"),
                "trainval": MockDataset("trainval"),
                "val": MockDataset("val"),
            }
        )
        assert det_train_dataset_adapter._get_subset_data("val", dataset) == "val"

        dataset = MockDatumDataset(
            {
                "train2017": MockDataset("train2017"),
                "val2017": MockDataset("val2017"),
                "test2017": MockDataset("test2017"),
            }
        )
        assert det_train_dataset_adapter._get_subset_data("val", dataset) == "val2017"

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

        instance_seg_otx_train_data = instance_seg_train_dataset_adapter.get_otx_dataset()
        instance_seg_otx_train_label_schema = instance_seg_train_dataset_adapter.get_label_schema()
        assert isinstance(instance_seg_otx_train_data, DatasetEntity)
        assert isinstance(instance_seg_otx_train_label_schema, LabelSchemaEntity)

        # In the test data, there is a empty_label image.
        # So, has_empty_label should be True
        has_empty_label = False
        for train_data in instance_seg_otx_train_data:
            if isinstance(train_data.annotation_scene, NullAnnotationSceneEntity):
                has_empty_label = True
        assert has_empty_label is True

        instance_seg_test_dataset_adapter = DetectionDatasetAdapter(
            task_type=task_type,
            test_data_roots=test_data_roots,
        )

        assert Subset.TESTING in instance_seg_test_dataset_adapter.dataset
        assert isinstance(instance_seg_test_dataset_adapter.get_otx_dataset(), DatasetEntity)
        assert isinstance(instance_seg_test_dataset_adapter.get_label_schema(), LabelSchemaEntity)
