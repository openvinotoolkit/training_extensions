# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

from otx.algorithms.classification.adapters.mmcls.datasets import OTXClsDataset
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.id import ID
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from otx.api.entities.subset import Subset
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


def load_test_dataset():
    """Helper to create test dataset"""

    def gen_image(resolution, x1, y1, x2, y2):
        w, h = resolution
        image = np.full([h, w, 3], fill_value=255, dtype=np.uint8)
        image[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w), :] = np.array([0, 128, 128], dtype=np.uint8)[
            None, None, :
        ]
        return image, Rectangle(x1=x1, y1=y1, x2=x2, y2=y2)

    images = [
        gen_image((640, 480), 0.0, 0.0, 0.5, 0.5),
        gen_image((640, 480), 0.5, 0.0, 1.0, 0.5),
        gen_image((640, 480), 0.0, 0.5, 0.5, 1.0),
        gen_image((640, 480), 0.5, 0.5, 1.0, 1.0),
    ]
    labels = [LabelEntity(name="rect", domain=Domain.DETECTION, id=ID("0"))]

    def get_image(i, subset):
        image, bbox = images[i]
        return DatasetItemEntity(
            media=Image(data=image),
            annotation_scene=AnnotationSceneEntity(
                annotations=[Annotation(bbox, labels=[ScoredLabel(label=labels[0])])],
                kind=AnnotationSceneKind.ANNOTATION,
            ),
            subset=subset,
        )

    items = [
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(2, Subset.TRAINING),
        get_image(3, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(2, Subset.TRAINING),
        get_image(3, Subset.TRAINING),
        get_image(0, Subset.TRAINING),
        get_image(1, Subset.TRAINING),
        get_image(0, Subset.VALIDATION),
        get_image(1, Subset.VALIDATION),
        get_image(2, Subset.VALIDATION),
        get_image(3, Subset.VALIDATION),
        get_image(0, Subset.TESTING),
        get_image(1, Subset.TESTING),
        get_image(2, Subset.TESTING),
        get_image(3, Subset.TESTING),
    ]
    return DatasetEntity(items), labels


class TestOTXClsDatasetInputParamsValidation:
    @e2e_pytest_unit
    def test_otx_classification_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTXClsDataset object initialization parameters validation

        <b>Input data:</b>
        OTXClsDataset object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTXClsDataset initialization parameter
        """
        dataset, labels_list = load_test_dataset()

        correct_values_dict = {
            "otx_dataset": dataset,
            "labels": labels_list,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "otx_dataset" parameter
            ("otx_dataset", unexpected_str),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested label
            ("labels", [labels_list[0], unexpected_str]),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTXClsDataset,
        )

    @e2e_pytest_unit
    def test_otx_classification_dataset_getitem_params_validation(self):
        """
        <b>Description:</b>
        Check OTXClsDataset object "__getitem__" method input parameters validation

        <b>Input data:</b>
        "idx" non-integer parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__getitem__" method
        """
        dataset, labels_list = load_test_dataset()
        otx_classification_dataset = OTXClsDataset(otx_dataset=dataset, labels=labels_list)
        with pytest.raises(ValueError):
            otx_classification_dataset.__getitem__(index="unexpected string")  # type: ignore
