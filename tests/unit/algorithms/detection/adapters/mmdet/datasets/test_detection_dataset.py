"""Unit Test for otx.algorithms.detection.adapters.mmdet.data.dataset."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
from otx.algorithms.common.utils.utils import is_xpu_available
import pytest

from otx.algorithms.detection.adapters.mmdet.datasets.dataset import OTXDetDataset, get_annotation_mmdet_format
from otx.api.entities.label import Domain
from otx.api.entities.model_template import TaskType
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.detection.test_helpers import (
    MockPipeline,
    generate_det_dataset,
)
from mmdet.core.mask.structures import BitmapMasks
import pycocotools.mask as mask_util

from otx.algorithms.detection.utils import create_detection_shapes, create_mask_shapes


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
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline, test_mode=False)
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
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline, test_mode=True)
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
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline)
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
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline)
        dataset.pipeline = MockPipeline()
        sample = dataset[0]
        if task_type == TaskType.DETECTION:
            results = [[np.random.rand(1, 5)]]
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            if is_xpu_available():
                pytest.skip("Subprocess failure in XPU environment")
            results = [
                (
                    [np.random.rand(1, 5)] * len(otx_dataset.get_labels()),
                    [[{"size": [sample["width"], sample["height"]], "counts": b"1"}]] * len(otx_dataset.get_labels()),
                )
            ]
        eval_results = dataset.evaluate(results, metric, logger)
        assert isinstance(eval_results, dict)
        assert metric in eval_results

    @e2e_pytest_unit
    def test_mask_evaluate(self) -> None:
        """Test evaluate method for instance segmentation"""
        if is_xpu_available():
            pytest.skip("Subprocess failure in XPU environment")
        otx_dataset, labels = self.dataset[TaskType.INSTANCE_SEGMENTATION]
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline)
        dataset.pipeline = MockPipeline()
        sample = dataset[0]

        num_classes = len(dataset.labels)
        anno = get_annotation_mmdet_format(sample["dataset_item"], dataset.labels, Domain.INSTANCE_SEGMENTATION)
        bboxes = anno["bboxes"]
        scores = np.random.random((len(bboxes), 1))
        bboxes = np.hstack((bboxes, scores))
        labels = anno["labels"]
        masks = mask_util.encode(np.full((28, 28, len(bboxes)), 1, dtype=np.uint8, order="F"))

        bbox_results = [bboxes[labels == i, :] for i in range(num_classes)]
        mask_results = [list(np.array(masks)[labels == i]) for i in range(num_classes)]
        results = [(bbox_results, mask_results)]
        eval_results = dataset.evaluate(results, "mAP", None)
        assert isinstance(eval_results, dict)
        assert eval_results["mAP"] >= 0.0

    @e2e_pytest_unit
    @pytest.mark.parametrize("use_ellipse_shapes", [True, False])
    def test_create_detection_shape(self, use_ellipse_shapes) -> None:
        """Test create_detection_shapes method"""
        otx_dataset, labels = self.dataset[TaskType.DETECTION]
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline)
        dataset.pipeline = MockPipeline()
        sample = dataset[0]
        h, w = sample["dataset_item"].height, sample["dataset_item"].width

        num_classes = len(dataset.labels)
        anno = get_annotation_mmdet_format(sample["dataset_item"], dataset.labels, Domain.DETECTION)
        bboxes = anno["bboxes"]
        scores = np.full((len(bboxes), 1), 0.5, dtype=np.float32)
        bboxes = np.hstack((bboxes, scores))
        labels = anno["labels"]
        pred_results = []
        for i in range(num_classes):
            bboxes_i = bboxes[labels == i, :]
            pred_results.append(bboxes_i)

        shapes = create_detection_shapes(
            pred_results,
            width=w,
            height=h,
            confidence_threshold=0.0,
            use_ellipse_shapes=use_ellipse_shapes,
            labels=dataset.labels,
        )
        assert len(shapes) > 0, "Shapes should be created for confidence_threshold=1.0"

        shapes = create_detection_shapes(
            pred_results,
            width=w,
            height=h,
            confidence_threshold=0.6,
            use_ellipse_shapes=use_ellipse_shapes,
            labels=dataset.labels,
        )
        assert len(shapes) == 0, "No shapes should be created for confidence_threshold=0.0"

    @e2e_pytest_unit
    @pytest.mark.parametrize("use_ellipse_shapes", [True, False])
    def test_create_mask_shape(self, use_ellipse_shapes) -> None:
        """Test create_mask_shapes method"""
        otx_dataset, labels = self.dataset[TaskType.INSTANCE_SEGMENTATION]
        dataset = OTXDetDataset(otx_dataset, labels, self.pipeline)
        dataset.pipeline = MockPipeline()
        sample = dataset[0]
        h, w = sample["dataset_item"].height, sample["dataset_item"].width

        num_classes = len(dataset.labels)
        anno = get_annotation_mmdet_format(sample["dataset_item"], dataset.labels, Domain.INSTANCE_SEGMENTATION)
        bboxes = anno["bboxes"]
        scores = np.full((len(bboxes), 1), 0.5, dtype=np.float32)
        bboxes = np.hstack((bboxes, scores))
        labels = anno["labels"]
        masks = mask_util.encode(np.full((28, 28, len(bboxes)), 1, dtype=np.uint8, order="F"))

        bbox_results = [bboxes[labels == i, :] for i in range(num_classes)]
        mask_results = [list(np.array(masks)[labels == i]) for i in range(num_classes)]
        pred_results = (bbox_results, mask_results)
        shapes = create_mask_shapes(
            pred_results,
            width=w,
            height=h,
            confidence_threshold=0.0,
            use_ellipse_shapes=use_ellipse_shapes,
            labels=dataset.labels,
        )
        assert len(shapes) > 0, "Shapes should be created for confidence_threshold=1.0"

        shapes = create_mask_shapes(
            pred_results,
            width=w,
            height=h,
            confidence_threshold=0.6,
            use_ellipse_shapes=use_ellipse_shapes,
            labels=dataset.labels,
        )
        assert len(shapes) == 0, "No shapes should be created for confidence_threshold=0.0"
