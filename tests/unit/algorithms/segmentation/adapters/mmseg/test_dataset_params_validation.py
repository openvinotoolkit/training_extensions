# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

from otx.algorithms.segmentation.adapters.mmseg.datasets.dataset import (
    OTXSegDataset,
    get_annotation_mmseg_format,
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
    return LabelEntity(name="test label", domain=Domain.SEGMENTATION)


def dataset_item():
    image = Image(data=np.random.randint(low=0, high=255, size=(10, 16, 3)))
    annotation = Annotation(shape=Rectangle.generate_full_box(), labels=[ScoredLabel(label_entity())])
    annotation_scene = AnnotationSceneEntity(annotations=[annotation], kind=AnnotationSceneKind.ANNOTATION)
    return DatasetItemEntity(media=image, annotation_scene=annotation_scene)


class TestOTXSegDatasetInputParamsValidation:
    @staticmethod
    def dataset():
        return OTXSegDataset(
            otx_dataset=DatasetEntity(),
            pipeline=[{"type": "LoadImageFromFile", "to_float32": True}],
            classes=["class_1", "class_2"],
        )

    @e2e_pytest_unit
    def test_otx_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTXSegDataset object initialization parameters validation

        <b>Input data:</b>
        OTXSegDataset object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTXSegDataset object initialization parameter
        """
        correct_values_dict = {
            "otx_dataset": DatasetEntity(),
            "pipeline": [{"type": "LoadImageFromFile", "to_float32": True}],
        }
        unexpected_str = "unexpected string"
        unexpected_int = 1
        unexpected_values = [
            # Unexpected string is specified as "otx_dataset" parameter
            ("otx_dataset", unexpected_str),
            # Unexpected integer is specified as "pipeline" parameter
            ("pipeline", unexpected_int),
            # Unexpected string is specified as nested pipeline
            ("pipeline", [{"config": 1}, unexpected_str]),
            # Unexpected string is specified as "classes" parameter
            ("classes", unexpected_str),
            # Unexpected string is specified as nested class
            ("classes", ["class_1", unexpected_int]),
            # Unexpected string is specified as "test_mode" parameter
            ("test_mode", unexpected_str),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTXSegDataset,
        )

    @e2e_pytest_unit
    def test_otx_dataset_filter_labels_params_validation(self):
        """
        <b>Description:</b>
        Check OTXSegDataset object "filter_labels" method input parameters validation

        <b>Input data:</b>
        OTXSegDataset object, "filter_labels" method unexpected parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "filter_labels" method
        """
        label = label_entity()
        dataset = self.dataset()
        correct_values_dict = {
            "all_labels": [label],
            "label_names": ["label_1", "label_2"],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "all_labels" parameter
            ("all_labels", unexpected_int),
            # Unexpected integer is specified as nested label
            ("all_labels", [label, unexpected_int]),
            # Unexpected integer is specified as "label_names" parameter
            ("label_names", unexpected_int),
            # Unexpected integer is specified as nested name
            ("label_names", ["label_1", unexpected_int]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=dataset.filter_labels,
        )

    @e2e_pytest_unit
    def test_otx_dataset_pre_pipeline_params_validation(self):
        """
        <b>Description:</b>
        Check OTXSegDataset object "pre_pipeline" method input parameters validation

        <b>Input data:</b>
        OTXSegDataset object, "results" unexpected type object

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
    def test_otx_dataset_prepare_train_img_params_validation(self):
        """
        <b>Description:</b>
        Check OTXSegDataset object "prepare_train_img" method input parameters validation

        <b>Input data:</b>
        OTXSegDataset object, "idx" non-integer type parameter

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
        Check OTXSegDataset object "prepare_test_img" method input parameters validation

        <b>Input data:</b>
        OTXSegDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "prepare_test_img" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.prepare_test_img(idx="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_dataset_get_ann_info_params_validation(self):
        """
        <b>Description:</b>
        Check OTXSegDataset object "get_ann_info" method input parameters validation

        <b>Input data:</b>
        OTXSegDataset object, "idx" non-integer type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_ann_info" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.get_ann_info(idx="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_otx_dataset_get_gt_seg_maps_params_validation(self):
        """
        <b>Description:</b>
        Check OTXSegDataset object "get_gt_seg_maps" method input parameters validation

        <b>Input data:</b>
        OTXSegDataset object, "efficient_test" non-bool type parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_gt_seg_maps" method
        """
        dataset = self.dataset()
        with pytest.raises(ValueError):
            dataset.get_gt_seg_maps(efficient_test="unexpected string")  # type: ignore


class TestMMDatasetFunctionsInputParamsValidation:
    @e2e_pytest_unit
    def test_get_annotation_mmseg_format_input_params_validation(self):
        """
        <b>Description:</b>
        Check "get_annotation_mmseg_format" function input parameters validation

        <b>Input data:</b>
        "get_annotation_mmseg_format" function unexpected-type input parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_annotation_mmseg_format" function
        """
        label = label_entity()
        correct_values_dict = {
            "dataset_item": dataset_item(),
            "labels": [label],
        }
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "dataset_item" parameter
            ("dataset_item", unexpected_int),
            # Unexpected integer is specified as "labels" parameter
            ("labels", unexpected_int),
            # Unexpected integer is specified as nested label
            ("labels", [label, unexpected_int]),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_annotation_mmseg_format,
        )
