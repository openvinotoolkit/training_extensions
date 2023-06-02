"""Unit-Test case for otx.core.data.adapter.segmentation_dataset_adapter."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import shutil
from pathlib import Path
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
    def setup_class(self) -> None:
        self.root_path = os.getcwd()
        task = "segmentation"

        self.task_type: TaskType = TASK_NAME_TO_TASK_TYPE[task]
        data_root_dict: dict = TASK_NAME_TO_DATA_ROOT[task]
        self.train_data_roots: str = os.path.join(self.root_path, data_root_dict["train"], "images")

        self.pseudo_mask_dir = Path(os.path.abspath(self.train_data_roots.replace("images", "detcon_mask")))

    def teardown_class(self) -> None:
        shutil.rmtree(self.pseudo_mask_dir, ignore_errors=True)

    @e2e_pytest_unit
    def test_import_dataset_create_all_masks(self, mocker):
        """Test _import_datasets when creating all masks.

        This test is for when all masks are not created and it is required to create masks.
        """
        shutil.rmtree(self.pseudo_mask_dir, ignore_errors=True)
        spy_create_pseudo_masks = mocker.spy(SelfSLSegmentationDatasetAdapter, "create_pseudo_masks")

        dataset_adapter = SelfSLSegmentationDatasetAdapter(
            task_type=self.task_type, train_data_roots=self.train_data_roots, pseudo_mask_dir=self.pseudo_mask_dir
        )

        spy_create_pseudo_masks.assert_called()
        assert spy_create_pseudo_masks.call_count == len(dataset_adapter.dataset[Subset.TRAINING])

    @e2e_pytest_unit
    @pytest.mark.parametrize("idx_remove", [1, 2, 3])
    def test_import_dataset_create_some_uncreated_masks(self, mocker, idx_remove: int):
        """Test _import_datasets when there are both uncreated and created masks.

        This test is for when there are both created and uncreated masks
        and it is required to either create or just load masks.
        In this test, remove a mask created before and check if `create_pseudo_masks` is called once.
        """
        shutil.rmtree(self.pseudo_mask_dir, ignore_errors=True)
        dataset_adapter = SelfSLSegmentationDatasetAdapter(
            task_type=self.task_type, train_data_roots=self.train_data_roots, pseudo_mask_dir=self.pseudo_mask_dir
        )
        assert os.path.isdir(self.pseudo_mask_dir)
        assert len(os.listdir(self.pseudo_mask_dir)) == 4

        # remove a mask
        os.remove(os.path.join(self.pseudo_mask_dir, f"000{idx_remove}.png"))
        spy_create_pseudo_masks = mocker.spy(SelfSLSegmentationDatasetAdapter, "create_pseudo_masks")

        _ = dataset_adapter._import_datasets(
            train_data_roots=self.train_data_roots, pseudo_mask_dir=self.pseudo_mask_dir
        )

        spy_create_pseudo_masks.assert_called()
        assert spy_create_pseudo_masks.call_count == 1

    @e2e_pytest_unit
    def test_import_dataset_just_load_masks(self, mocker):
        """Test _import_datasets when just loading all masks."""
        spy_create_pseudo_masks = mocker.spy(SelfSLSegmentationDatasetAdapter, "create_pseudo_masks")

        _ = SelfSLSegmentationDatasetAdapter(
            task_type=self.task_type, train_data_roots=self.train_data_roots, pseudo_mask_dir=self.pseudo_mask_dir
        )

        spy_create_pseudo_masks.assert_not_called()

    @e2e_pytest_unit
    @pytest.mark.xfail
    def test_get_otx_dataset_without_skipping_background(self):
        """Test get_otx_dataset without skipping background.

        TODO (sungchul): don't skip background class in get_otx_dataset
        """
        assert 0

    @e2e_pytest_unit
    def test_create_pseudo_masks(self, mocker):
        """Test create_pseudo_masks."""
        mocker.patch("otx.core.data.adapter.segmentation_dataset_adapter.os.makedirs")
        mocker.patch("otx.core.data.adapter.segmentation_dataset_adapter.cv2.imwrite")
        dataset_adapter = SelfSLSegmentationDatasetAdapter(
            task_type=self.task_type, train_data_roots=self.train_data_roots, pseudo_mask_dir=self.pseudo_mask_dir
        )

        pseudo_mask = dataset_adapter.create_pseudo_masks(img=np.ones((2, 2)), pseudo_mask_path="")

        assert type(pseudo_mask) == np.ndarray
