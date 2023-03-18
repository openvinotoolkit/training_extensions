"""Unit Test for otx.algorithms.detection.adapters.mmdet.data.dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

from otx.algorithms.detection.adapters.mmdet.datasets.dataset import OTXDetDataset
from otx.api.entities.label import Domain
from otx.api.entities.model_template import TaskType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    MockPipeline,
    generate_det_dataset,
)


class TestOTXDetDataset:
    """
    Test OTXDetDataset class.
    1. Test _DataInfoProxy
    2. Test prepare_train_img
    3. Test prepare_test_img
    4. Test get_ann_info
    5. Test evaluate
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.dataset = dict()
        for task_type in [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION]:
            self.dataset[task_type] = generate_det_dataset(task_type=task_type)
        self.pipeline = []

    @e2e_pytest_unit
    @pytest.mark.parametrize("task_type", [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION])
    def test_DataInfoProxy(self, task_type):
        """Test _DataInfoProxy Class."""
        otx_dataset, labels = self.dataset[task_type]
        proxy = OTXDetDataset._DataInfoProxy(otx_dataset, labels)
        sample = proxy[0]
        assert "dataset_item" in sample
        assert "width" in sample
        assert "height" in sample
        assert "index" in sample
        assert "ann_info" in sample
        assert "ignored_labels" in sample

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    def test_prepare_train_img(self, task_type, domain) -> None:
        """Test prepare_train_img method"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline, domain, test_mode=False)
        img = dataset.prepare_train_img(0)
        assert isinstance(img, dict)
        assert "dataset_item" in img
        assert "bbox_fields" in img
        assert "mask_fields" in img
        assert "seg_fields" in img

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    def test_prepare_test_img(self, task_type, domain) -> None:
        """Test prepare_test_img method"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline, domain, test_mode=True)
        img = dataset.prepare_test_img(0)
        assert isinstance(img, dict)
        assert "dataset_item" in img
        assert "bbox_fields" in img
        assert "mask_fields" in img
        assert "seg_fields" in img

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    def test_get_ann_info(self, task_type, domain) -> None:
        """Test get_ann_info method"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline, domain)
        dataset.pipeline = MockPipeline()
        ann_info = dataset.get_ann_info(0)
        assert isinstance(ann_info, dict)
        assert "bboxes" in ann_info
        assert "masks" in ann_info
        assert "labels" in ann_info

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    @pytest.mark.parametrize("metric", ["mAP"])
    @pytest.mark.parametrize("logger", ["silent", None])
    def test_evaluate(self, task_type, domain, metric, logger) -> None:
        """Test evaluate method for detection and instance segmentation"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline, domain)
        dataset.pipeline = MockPipeline()
        sample = dataset[0]
        if task_type == TaskType.DETECTION:
            results = [[np.random.rand(1, 5)]]
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            results = [
                (
                    [np.random.rand(1, 5)],
                    [[{"size": [sample["width"], sample["height"]], "counts": "some counts"}]],
                )
            ]
        eval_results = dataset.evaluate(results, metric, logger)
        assert isinstance(eval_results, dict)
        assert metric in eval_results
