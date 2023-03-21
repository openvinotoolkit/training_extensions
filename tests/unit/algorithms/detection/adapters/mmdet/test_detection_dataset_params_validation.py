# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.algorithms.detection.adapters.mmdet.datasets.dataset import (
    OTXDetDataset,
    get_annotation_mmdet_format,
)
from otx.api.entities.annotation import (
    Annotation,
    AnnotationSceneEntity,
    AnnotationSceneKind,
)
from otx.api.entities.dataset_item import DatasetItemEntity
from otx.api.entities.datasets import DatasetEntity
from otx.api.entities.image import Image
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.scored_label import ScoredLabel
from otx.api.entities.shapes.rectangle import Rectangle
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


def label_entity():
    return LabelEntity(name="test label", domain=Domain.DETECTION)


def dataset_item():
    image = Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)))
    annotation = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(label_entity())])
    annotation_scene = AnnotationSceneEntity(annotations=[annotation], kind=AnnotationSceneKind.ANNOTATION)
    return DatasetItemEntity(media=image, annotation_scene=annotation_scene)


class TestMMDatasetFunctionsInputParamsValidation:
    @e2e_pytest_unit
    def test_get_annotation_mmdet_format_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_annotation_mmdet_format" function input parameters validation

        <b>Input data:</b>
        "get_annotation_mmdet_format" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_annotation_mmdet_format" function
        """
        label = label_entity()
        correct_values_dict = {
            "dataset_item": dataset_item(),
            "labels": [label],
            "domain": Domain.DETECTION,
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "dataset_item" parameter
            ("dataset_item", unexpected_int),
            # Unexpected integer is specified as "labels" parameter
            ("labels", unexpected_int),
            # Unexpected integer is specified as nested label
            ("labels", [label, unexpected_int]),
            # Unexpected integer is specified as "domain" parameter
            ("domain", unexpected_int),
            # Unexpected string is specified as "min_size" parameter
            ("min_size", "unexpected string"),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_annotation_mmdet_format,
        )


class TestOTXDetDatasetInputParamsValidation:
    @staticmethod
    def dataset():
        pipeline = [{"type": "LoadImageFromFile", "to_float32": True}]
        return OTXDetDataset(
            otx_dataset=DatasetEntity(),
            labels=[label_entity()],
            pipeline=pipeline,
            test_mode=True,
            domain=Domain.DETECTION,
        )

    @e2e_pytest_unit
    def test_otx_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTXDetDataset object initialization parameters validation

        <b>Input data:</b>
        OTXDetDataset object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTXDetDataset object initialization parameter
        """
        label = label_entity()

        correct_values_dict = {
            "otx_dataset": DatasetEntity(),
            "labels": [label],
            "pipeline": [{"type": "LoadImageFromFile", "to_float32": True}],
            "domain": Domain.DETECTION,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "otx_dataset" parameter
            ("otx_dataset", unexpected_str),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested label
            ("labels", [label, unexpected_str]),
            # Unexpected integer is specified as "pipeline" parameter
            ("pipeline", 1),
            # Unexpected string is specified as nested pipeline
            ("pipeline", [{"config": 1}, unexpected_str]),
            # Unexpected string is specified as "domain" parameter
            ("domain", unexpected_str),
            # Unexpected string is specified as "test_mode" parameter
            ("test_mode", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTXDetDataset,
        )

    @e2e_pytest_unit
    def test_otx_dataset_prepare_train_img_params_validation(self):
        """
        <b>Description:</b>
        Check OTXDetDataset object "prepare_train_img" method input parameters validation

        <b>Input data:</b>
        OTXDetDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "prepare_train_img" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.prepare_train_img(idx="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_dataset_prepare_test_img_params_validation(self):
        """
        <b>Description:</b>
        Check OTXDetDataset object "prepare_test_img" method input parameters validation

        <b>Input data:</b>
        OTXDetDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "prepare_test_img" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.prepare_test_img(idx="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_dataset_pre_pipeline_params_validation(self):
        """
        <b>Description:</b>
        Check OTXDetDataset object "pre_pipeline" method input parameters validation

        <b>Input data:</b>
        OTXDetDataset object, "results" unexpected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "pre_pipeline" method
        """
        dataset = self.dataset()
        unexpected_int = 1
        for unexpected_value in [
            # Unexpected integer is specified as "results" parameter
            unexpected_int,
            # Unexpected integer is specified as "results" dictionary key
            {"result_1": "some results", unexpected_int: "unexpected results"},
        ]:
            with pytest.raises(ValueError):
                dataset.pre_pipeline(results=unexpected_value)

    @e2e_pytest_unit
    def test_otx_dataset_get_ann_info_params_validation(self):
        """
        <b>Description:</b>
        Check OTXDetDataset object "get_ann_info" method input parameters validation

        <b>Input data:</b>
        OTXDetDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_ann_info" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.get_ann_info(idx="unexpected string")  # type: ignore
