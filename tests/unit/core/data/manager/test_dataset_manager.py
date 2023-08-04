"""Unit-Test case for otx.core.data.manager."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import shutil
from typing import List
from tempfile import TemporaryDirectory

import pytest

from datumaro.components.dataset import DatasetSubset
from otx.cli.manager.config_manager import TASK_TYPE_TO_SUPPORTED_FORMAT
from otx.core.data.manager.dataset_manager import DatasetManager
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.core.data.test_helpers import (
    generate_datumaro_dataset,
    generate_datumaro_dataset_item,
)

AVAILABLE_TASKS = ["classification", "detection", "segmentation"]
AVAILABLE_SUBSETS = ["train", "val"]
AVAILABLE_DATA_ROOTS = [
    "tests/assets/classification_dataset",
    "tests/assets/car_tree_bug",
    "tests/assets/cityscapes_dataset/dataset",
    "tests/assets/anomaly/hazelnut",
    "tests/assets/cvat_dataset/action_classification/train",
]

DATA_ROOTS2FORMAT = {
    "tests/assets/classification_dataset": "imagenet",
    "tests/assets/car_tree_bug": "coco",
    "tests/assets/cityscapes_dataset/dataset": "cityscapes",
}


class TestOTXDatasetManager:
    def setup_method(self) -> None:
        self.dataset = {}
        for subset in AVAILABLE_SUBSETS:
            self.dataset[subset] = {}
            for task in AVAILABLE_TASKS:
                self.dataset[subset][task] = generate_datumaro_dataset(subsets=[subset], task=task)

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", AVAILABLE_TASKS)
    @pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
    def test_get_train_dataset(self, task: List[str], subset: List[str]):
        if subset == "val":
            with pytest.raises(ValueError, match="Can't find training data."):
                DatasetManager.get_train_dataset(self.dataset[subset][task])
        else:
            train_dataset = DatasetManager.get_train_dataset(self.dataset[subset][task])
            assert isinstance(train_dataset, DatasetSubset)

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", AVAILABLE_TASKS)
    @pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
    def test_get_val_dataset(self, task: List[str], subset: List[str]):
        if subset == "train":
            assert DatasetManager.get_val_dataset(self.dataset[subset][task]) is None
        else:
            val_dataset = DatasetManager.get_val_dataset(self.dataset[subset][task])
            assert isinstance(val_dataset, DatasetSubset)

    @e2e_pytest_unit
    @pytest.mark.parametrize("data_root", AVAILABLE_DATA_ROOTS)
    def test_get_data_format(self, data_root: str):
        assert isinstance(DatasetManager.get_data_format(data_root), str)

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", AVAILABLE_TASKS)
    @pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
    def test_get_image_path(self, task, subset):
        random_data = DatasetManager.get_image_path(
            generate_datumaro_dataset_item(item_id="0", subset=subset, task=task)
        )
        assert random_data is None

        with TemporaryDirectory() as temp_dir:
            random_data = DatasetManager.get_image_path(
                generate_datumaro_dataset_item(item_id="0", subset=subset, task=task, temp_dir=temp_dir)
            )
            assert random_data is not None

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", AVAILABLE_TASKS)
    @pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
    def test_export_dataset(self, task, subset, tmp_dir_path):
        data_format = TASK_TYPE_TO_SUPPORTED_FORMAT[task.upper()][0]
        DatasetManager.export_dataset(self.dataset[subset][task], tmp_dir_path, data_format, save_media=False)
        shutil.rmtree(tmp_dir_path)

    @e2e_pytest_unit
    @pytest.mark.parametrize("data_root", AVAILABLE_DATA_ROOTS[:3])
    def test_import_dataset(self, data_root):
        data_format = DATA_ROOTS2FORMAT[data_root]
        assert DatasetManager.import_dataset(data_root, data_format=data_format) is not None

    # TODO: Currently, direct annotation only supports COCO format
    @e2e_pytest_unit
    @pytest.mark.parametrize("data_root", [AVAILABLE_DATA_ROOTS[1]])
    def test_import_dataset_with_direct_annotation(self, data_root):
        data_format = DATA_ROOTS2FORMAT[data_root]
        assert DatasetManager.import_dataset(data_root, data_format=data_format) is not None

        ann_files = "tests/assets/car_tree_bug/annotations/instances_train_5_imgs.json"
        train_dataset = DatasetManager.import_dataset(ann_files, data_format=data_format, subset="train")
        assert train_dataset.get_subset("train_5_imgs").get_annotated_items() == 5

    @e2e_pytest_unit
    @pytest.mark.parametrize("task", AVAILABLE_TASKS)
    @pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
    def test_auto_split(self, task, subset):
        dataset = DatasetManager.auto_split(
            task=task, dataset=self.dataset[subset][task], split_ratio=[("train", 0.8), ("val", 0.2)]
        )
        assert dataset is not None
