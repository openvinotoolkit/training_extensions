# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import pytest

from otx.algorithms.classification.utils import (
    ClassificationDatasetAdapter,
    generate_label_schema,
    get_multihead_class_info,
)
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.api.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)


class TestClassificationDatasetAdapterInputParamsValidation:
    @e2e_pytest_unit
    def test_classification_dataset_adapter_init_params_validation(self):
        """
        <b>Description:</b>
        Check ClassificationDatasetAdapter object initialization parameters validation

        <b>Input data:</b>
        ClassificationDatasetAdapter object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        ClassificationDatasetAdapter initialization parameter
        """
        correct_values_dict = {}
        unexpected_int = 1
        unexpected_values = [
            # Unexpected integer is specified as "train_ann_file" parameter
            ("train_ann_file", unexpected_int),
            # Empty string is specified as "train_ann_file" parameter
            ("train_ann_file", ""),
            # Path with null character is specified as "train_ann_file" parameter
            ("train_ann_file", "./\0fake_data.json"),
            # Path with non-printable character is specified as "train_ann_file" parameter
            ("train_ann_file", "./\nfake_data.json"),
            # Unexpected integer is specified as "train_data_root" parameter
            ("train_data_root", unexpected_int),
            # Empty string is specified as "train_data_root" parameter
            ("train_data_root", ""),
            # Path with null character is specified as "train_data_root" parameter
            ("train_data_root", "./\0null_char"),
            # Path with non-printable character is specified as "train_data_root" parameter
            ("train_data_root", "./\non_printable_char"),
            # Unexpected integer is specified as "val_ann_file" parameter
            ("val_ann_file", unexpected_int),
            # Empty string is specified as "val_ann_file" parameter
            ("val_ann_file", ""),
            # Path with null character is specified as "val_ann_file" parameter
            ("val_ann_file", "./\0fake_data.json"),
            # Path with non-printable character is specified as "val_ann_file" parameter
            ("val_ann_file", "./\nfake_data.json"),
            # Unexpected integer is specified as "val_data_root" parameter
            ("val_data_root", unexpected_int),
            # Empty string is specified as "val_data_root" parameter
            ("val_data_root", ""),
            # Path with null character is specified as "val_data_root" parameter
            ("val_data_root", "./\0null_char"),
            # Path with non-printable character is specified as "val_data_root" parameter
            ("val_data_root", "./\non_printable_char"),
            # Unexpected integer is specified as "test_ann_file" parameter
            ("test_ann_file", unexpected_int),
            # Empty string is specified as "test_ann_file" parameter
            ("test_ann_file", ""),
            # Path with null character is specified as "test_ann_file" parameter
            ("test_ann_file", "./\0fake_data.json"),
            # Path with non-printable character is specified as "test_ann_file" parameter
            ("test_ann_file", "./\nfake_data.json"),
            # Unexpected integer is specified as "test_data_root" parameter
            ("test_data_root", unexpected_int),
            # Empty string is specified as "test_data_root" parameter
            ("test_data_root", ""),
            # Path with null character is specified as "test_data_root" parameter
            ("test_data_root", "./\0null_char"),
            # Path with non-printable character is specified as "test_data_root" parameter
            ("test_data_root", "./\non_printable_char"),
        ]
        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=ClassificationDatasetAdapter,
        )


class TestUtilsFunctionsParamsValidation:
    @e2e_pytest_unit
    def test_generate_label_schema_params_validation(self):
        """
        <b>Description:</b>
        Check "get_multilabel_predictions" function input parameters validation

        <b>Input data:</b>
        "get_multilabel_predictions" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_multilabel_predictions" function
        """
        labels_list = [LabelEntity(name="rect", domain=Domain.DETECTION, id=ID("0"))]

        correct_values_dict = {
            "not_empty_labels": labels_list,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "not_empty_labels" parameter
            ("not_empty_labels", unexpected_str),
            # Unexpected string is specified as nested non_empty_label
            ("not_empty_labels", [labels_list[0], unexpected_str]),
            # Unexpected string is specified as "multilabel" parameter
            ("multilabel", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=generate_label_schema,
        )

    @e2e_pytest_unit
    def test_get_multihead_class_info_params_validation(self):
        """
        <b>Description:</b>
        Check "get_multihead_class_info" function input parameters validation

        <b>Input data:</b>
        "label_schema" non-LabelSchemaEntity parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_multihead_class_info" function
        """
        with pytest.raises(ValueError):
            get_multihead_class_info(label_schema=1)  # type: ignore
