# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.detection.adapters.mmdet.data import MPADetDataset
from otx.algorithms.detection.utils import generate_label_schema
from otx.api.entities.annotation import AnnotationSceneEntity, AnnotationSceneKind
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain
from otx.api.entities.model_template import TaskType, task_type_to_label_domain
from otx.api.utils.shape_factory import ShapeFactory
from tests.test_helpers import generate_random_annotated_image
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


def create_det_dataset(task_type, number_of_images=1):
    classes = ("rectangle", "ellipse", "triangle")
    label_schema = generate_label_schema(classes, task_type_to_label_domain(task_type))

    items = []
    for _ in range(number_of_images):
        image_numpy, annos = generate_random_annotated_image(
            image_width=640,
            image_height=480,
            labels=label_schema.get_labels(False),
        )
        # Convert shapes according to task
        for anno in annos:
            if task_type == TaskType.DETECTION:
                anno.shape = ShapeFactory.shape_as_rectangle(anno.shape)
            elif task_type == TaskType.INSTANCE_SEGMENTATION:
                anno.shape = ShapeFactory.shape_as_polygon(anno.shape)
        image = Image(data=image_numpy)
        annotation_scene = AnnotationSceneEntity(kind=AnnotationSceneKind.ANNOTATION, annotations=annos)
        items.append(DatasetItemEntity(media=image, annotation_scene=annotation_scene))
    dataset = DatasetEntity(items)
    return dataset, dataset.get_labels()


class TestOTXDetDataset:
    """Test OTXDetDataset functionality"""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.dataset = dict()
        for task_type in [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION]:
            self.dataset[task_type] = create_det_dataset(task_type=task_type)
        self.pipeline = []

    @e2e_pytest_unit
    @pytest.mark.parametrize("task_type", [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION])
    def test_otx_detection_dataset_init_params_validation(self, task_type):
        """Test OTXDetDataset initialization parameters validation"""
        otx_dataset, labels = self.dataset[task_type]
        correct_values_dict = {
            "otx_dataset": otx_dataset,
            "labels": labels,
        }

        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "otx_dataset" parameter
            ("otx_dataset", unexpected_str),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested label
            ("labels", [labels[0], unexpected_str]),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=MPADetDataset,
        )

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    def test_prepare_train_img(self, task_type, domain) -> None:
        """Test prepare_train_img method"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = MPADetDataset(otx_dataset, labels, self.pipeline, domain)
        img = dataset.prepare_train_img(0)
        assert isinstance(img, dict)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    def test_prepare_train_img_out_of_index(self, task_type, domain) -> None:
        """Test prepare_train_img method for out of index"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = MPADetDataset(otx_dataset, labels, self.pipeline, domain)
        out_of_range_index = len(otx_dataset) + 1
        with pytest.raises(IndexError):
            dataset.prepare_train_img(out_of_range_index)

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    def test_prepare_test_img(self, task_type, domain) -> None:
        """Test prepare_test_img method"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = MPADetDataset(otx_dataset, labels, self.pipeline, domain)
        img = dataset.prepare_test_img(0)
        assert isinstance(img, dict)

    @e2e_pytest_unit
    def test_pre_pipeline(self) -> None:
        """Test pre_pipeline method"""
        results = dict()
        MPADetDataset.pre_pipeline(results)
        assert "bbox_fields" in results
        assert "mask_fields" in results
        assert "seg_fields" in results

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    def test_get_ann_info(self, task_type, domain) -> None:
        """test get_ann_info method"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = MPADetDataset(otx_dataset, labels, self.pipeline, domain)
        ann_info = dataset.get_ann_info(0)
        assert isinstance(ann_info, dict)
        assert "bboxes" in ann_info
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
        dataset = MPADetDataset(otx_dataset, labels, self.pipeline, domain)
        if task_type == TaskType.DETECTION:
            results = [[np.array([[0, 0, 32, 24, 0.55], [0, 0, 32, 24, 0.55]], dtype=np.float32)]]
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            results = [
                (
                    [np.array([[8, 5, 10, 20, 0.90]], dtype=np.float32)],
                    [[{"size": [640, 480], "counts": "some counts"}]],
                )
            ]
        eval_results = dataset.evaluate(results, metric, logger)
        assert isinstance(eval_results, dict)
        assert metric in eval_results

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    @pytest.mark.parametrize("metric", ["accuracy-top-1", "mDice", 123456789])
    def test_invalid_param_evaluate(self, task_type, domain, metric) -> None:
        """Test evaluate method for detection and instance segmentation"""
        otx_dataset, labels = self.dataset[task_type]
        dataset = MPADetDataset(otx_dataset, labels, self.pipeline, domain)
        if task_type == TaskType.DETECTION:
            results = [[np.array([[0, 0, 32, 24, 0.55], [0, 0, 32, 24, 0.55]], dtype=np.float32)]]
        elif task_type == TaskType.INSTANCE_SEGMENTATION:
            results = [
                (
                    [np.array([[8, 5, 10, 20, 0.90]], dtype=np.float32)],
                    [[{"size": [640, 480], "counts": "some counts"}]],
                )
            ]
        with pytest.raises(KeyError):
            dataset.evaluate(results, metric=metric)
