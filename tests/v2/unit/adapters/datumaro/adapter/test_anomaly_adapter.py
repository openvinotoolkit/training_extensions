"""Unit-Test case for otx.core.data.adapter.anomaly_dataset_adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from pathlib import Path
from typing import Dict, TYPE_CHECKING

from otx.v2.adapters.datumaro.adapter.anomaly_dataset_adapter import (
    AnomalyClassificationDatasetAdapter,
    AnomalyDetectionDatasetAdapter,
    AnomalySegmentationDatasetAdapter,
)
from otx.v2.api.entities.datasets import DatasetEntity
from otx.v2.api.entities.subset import Subset

from tests.v2.unit.adapters.datumaro.test_helpers import (
    TASK_NAME_TO_DATA_ROOT,
    TASK_NAME_TO_TASK_TYPE,
)

if TYPE_CHECKING:
    from otx.v2.api.entities.task_type import TaskType


class TestOTXAnomalyClassificationDatasetAdapter:
    def setup_method(self) -> None:
        self.root_path = Path.cwd()
        task = "anomaly_classification"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = str(self.root_path / data_root_dict["train"])
        self.val_data_roots: str = str(self.root_path / data_root_dict["val"])
        self.test_data_roots: str = str(self.root_path / data_root_dict["test"])

        self.train_dataset_adapter = AnomalyClassificationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
        )

        self.test_dataset_adapter = AnomalyClassificationDatasetAdapter(
            task_type=self.task_type,
            test_data_roots=self.test_data_roots,
        )

    def test_init(self) -> None:
        assert Subset.TRAINING in self.train_dataset_adapter.dataset
        assert Subset.VALIDATION in self.train_dataset_adapter.dataset
        assert Subset.TESTING in self.test_dataset_adapter.dataset

    def test_get_otx_dataset(self) -> None:
        assert isinstance(self.train_dataset_adapter.get_otx_dataset(), Dict)
        assert isinstance(self.test_dataset_adapter.get_otx_dataset(), Dict)


class TestOTXAnomalyDetectionDatasetAdapter:
    def setup_method(self) -> None:
        self.root_path = Path.cwd()
        task = "anomaly_detection"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = str(self.root_path / data_root_dict["train"])
        self.val_data_roots: str = str(self.root_path / data_root_dict["val"])
        self.test_data_roots: str = str(self.root_path / data_root_dict["test"])

        self.train_dataset_adapter = AnomalyDetectionDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
        )

        self.test_dataset_adapter = AnomalyDetectionDatasetAdapter(
            task_type=self.task_type,
            test_data_roots=self.test_data_roots,
        )

    def test_init(self) -> None:
        assert Subset.TRAINING in self.train_dataset_adapter.dataset
        assert Subset.VALIDATION in self.train_dataset_adapter.dataset
        assert Subset.TESTING in self.test_dataset_adapter.dataset

    def test_get_otx_dataset(self) -> None:
        assert isinstance(self.train_dataset_adapter.get_otx_dataset(), Dict)
        assert isinstance(self.test_dataset_adapter.get_otx_dataset(), Dict)


class TestOTXAnomalySegmentationDatasetAdapter:
    def setup_method(self) -> None:
        self.root_path = Path.cwd()
        task = "anomaly_segmentation"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]

        self.train_data_roots: str = str(self.root_path / data_root_dict["train"])
        self.val_data_roots: str = str(self.root_path / data_root_dict["val"])
        self.test_data_roots: str = str(self.root_path / data_root_dict["test"])

        self.train_dataset_adapter = AnomalySegmentationDatasetAdapter(
            task_type=self.task_type,
            train_data_roots=self.train_data_roots,
            val_data_roots=self.val_data_roots,
        )

        self.test_dataset_adapter = AnomalySegmentationDatasetAdapter(
            task_type=self.task_type,
            test_data_roots=self.test_data_roots,
        )

    def test_init(self) -> None:
        assert Subset.TRAINING in self.train_dataset_adapter.dataset
        assert Subset.VALIDATION in self.train_dataset_adapter.dataset
        assert Subset.TESTING in self.test_dataset_adapter.dataset

    def test_get_otx_dataset(self) -> None:
        assert isinstance(self.train_dataset_adapter.get_otx_dataset(), Dict)
        assert isinstance(self.test_dataset_adapter.get_otx_dataset(), Dict)
