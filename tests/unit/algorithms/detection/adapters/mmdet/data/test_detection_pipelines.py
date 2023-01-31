# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.detection.adapters.mmdet.data import (
    LoadAnnotationFromOTXDataset,
    LoadImageFromOTXDataset,
)
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


def otx_to_mmdet_converter(item, labels):
    results = dict(
        dataset_item=item,
        width=item.width,
        height=item.height,
        index=0,
        ann_info=dict(label_list=labels),
    )
    return results


class TestLoadImageFromOTXDataset:
    """Test class for LoadImageFromOTXDataset"""

    @e2e_pytest_unit
    @pytest.mark.parametrize("task_type", [TaskType.DETECTION, TaskType.INSTANCE_SEGMENTATION])
    @pytest.mark.parametrize("to_float32", [False, True])
    def test_load_image_from_otx_dataset_call(self, task_type, to_float32):
        """Test that the pipeline is called without errors"""
        otx_dataset, labels = create_det_dataset(task_type=task_type)
        pipeline = LoadImageFromOTXDataset(to_float32)
        results = otx_to_mmdet_converter(otx_dataset[0], labels)
        pipeline(results)
        assert "filename" in results
        assert "ori_filename" in results
        assert "img" in results
        assert "img_shape" in results
        assert "ori_shape" in results
        assert "pad_shape" in results
        assert "img_norm_cfg" in results
        assert "img_fields" in results
        assert isinstance(results["img"], np.ndarray)


class TestLoadAnnotationFromOTXDataset:
    """Test class for LoadAnnotationFromOTXDataset"""

    @e2e_pytest_unit
    @pytest.mark.parametrize(
        "task_type, domain",
        [(TaskType.DETECTION, Domain.DETECTION), (TaskType.INSTANCE_SEGMENTATION, Domain.INSTANCE_SEGMENTATION)],
    )
    @pytest.mark.parametrize("with_mask", [False, True])
    def test_load_image_from_otx_dataset_call(self, task_type, domain, with_mask):
        """Test that the pipeline is called without errors"""
        otx_dataset, labels = create_det_dataset(task_type=task_type)
        pipeline = LoadAnnotationFromOTXDataset(-1, with_mask=with_mask, domain=domain)
        results = otx_to_mmdet_converter(otx_dataset[0], labels)
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        pipeline(results)
        assert "gt_labels" in results
        if with_mask:
            assert "gt_masks" in results
        else:
            assert "gt_bboxes" in results
