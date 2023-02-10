"""Unit-Test case for otx.core.data.manager."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from otx.core.data.manager.dataset_manager import DatasetManager
from tests.test_helpers import generate_datumaro_dataset
from tests.test_suite.e2e_test_system import e2e_pytest_unit

AVAILABLE_TASKS=["classification", "detection", "segmentation"]
AVAILABLE_SUBSETS=["train", "val"]

class TestOTXDatasetManager:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.dataset = {}
        for subset in AVAILABLE_SUBSETS:
            self.dataset[subset] = {}
            for task in AVAILABLE_TASKS:
                self.dataset[subset][task] = generate_datumaro_dataset(
                    subsets=[subset],
                    task=task
                )
    
    @pytest.mark.parametrize("task", AVAILABLE_TASKS)
    @e2e_pytest_unit
    def test_get_train_dataset(self):
        train_dataset = DatasetManager.get_train_dataset(
            self.dataset
        )