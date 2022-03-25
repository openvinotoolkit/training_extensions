import numpy as np
import pytest
from ote_sdk.entities.label import LabelEntity, Domain
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_unit
from ote_sdk.tests.parameters_validation.validation_helper import (
    check_value_error_exception_raised,
)
from torchreid_tasks.utils import (
    ClassificationDatasetAdapter,
    active_score_from_probs,
    OTEClassificationDataset,
    generate_label_schema,
    get_actmap,
    get_multiclass_predictions,
    get_task_class,
    reload_hyper_parameters,
    preprocess_features_for_actmap,
    set_values_as_default,
    sigmoid_numpy,
    softmax_numpy,
    get_multilabel_predictions,
    force_fp32,
    get_multihead_class_info,
    get_hierarchical_predictions,
)

from helpers import load_test_dataset


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


class TestOTEClassificationDatasetInputParamsValidation:
    @e2e_pytest_unit
    def test_ote_classification_dataset_init_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationDataset object initialization parameters validation

        <b>Input data:</b>
        OTEClassificationDataset object initialization parameters with unexpected type

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        OTEClassificationDataset initialization parameter
        """
        dataset, labels_list = load_test_dataset()

        correct_values_dict = {
            "ote_dataset": dataset,
            "labels": labels_list,
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "ote_dataset" parameter
            ("ote_dataset", unexpected_str),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested label
            ("labels", [labels_list[0], unexpected_str]),
            # Unexpected string is specified as "multilabel" parameter
            ("multilabel", unexpected_str),
            # Unexpected string is specified as "hierarchical" parameter
            ("hierarchical", unexpected_str),
            # Unexpected string is specified as "mixed_cls_heads_info" parameter
            ("mixed_cls_heads_info", unexpected_str),
            # Unexpected string is specified as "keep_empty_label" parameter
            ("keep_empty_label", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=OTEClassificationDataset,
        )

    @e2e_pytest_unit
    def test_ote_classification_dataset_getitem_params_validation(self):
        """
        <b>Description:</b>
        Check OTEClassificationDataset object "__getitem__" method input parameters validation

        <b>Input data:</b>
        "idx" non-integer parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "__getitem__" method
        """
        dataset, labels_list = load_test_dataset()
        ote_classification_dataset = OTEClassificationDataset(
            ote_dataset=dataset,
            labels=labels_list
        )
        with pytest.raises(ValueError):
            ote_classification_dataset.__getitem__(idx="unexpected string")  # type: ignore


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
        dataset, labels_list = load_test_dataset()

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

    @e2e_pytest_unit
    def test_get_task_class_params_validation(self):
        """
        <b>Description:</b>
        Check "get_task_class" function input parameters validation

        <b>Input data:</b>
        "path" non-string parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_task_class" function
        """
        with pytest.raises(ValueError):
            get_task_class(path=1)  # type: ignore

    @e2e_pytest_unit
    def test_reload_hyper_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "reload_hyper_parameters" function input parameters validation

        <b>Input data:</b>
        "model_template" non-ModelTemplate parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "reload_hyper_parameters" function
        """
        with pytest.raises(ValueError):
            reload_hyper_parameters(model_template="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_set_values_as_default_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "set_values_as_default" function input parameters validation

        <b>Input data:</b>
        "parameters" non-ModelTemplate parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "set_values_as_default" function
        """
        with pytest.raises(ValueError):
            set_values_as_default(parameters="unexpected string")  # type: ignore

    @e2e_pytest_unit
    def test_force_fp32_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "force_fp32" function input parameters validation

        <b>Input data:</b>
        "model" non-Module parameter

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "force_fp32" function
        """
        with pytest.raises(ValueError):
            with force_fp32(model="unexpected string"):  # type: ignore
                pass

    @e2e_pytest_unit
    def test_preprocess_features_for_actmap_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "preprocess_features_for_actmap" function input parameters validation

        <b>Input data:</b>
        "features" non-expected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "preprocess_features_for_actmap" function
        """
        with pytest.raises(ValueError):
            preprocess_features_for_actmap(features=None)  # type: ignore

    @e2e_pytest_unit
    def test_get_actmap_params_validation(self):
        """
        <b>Description:</b>
        Check "get_actmap" function input parameters validation

        <b>Input data:</b>
        "get_actmap" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_actmap" function
        """
        correct_values_dict = {
            "features": ["some", "features"],
            "output_res": ("iterable", "object")
        }
        unexpected_values = [
            # Unexpected dictionary is specified as "features" parameter
            ("features", None),
            # Unexpected dictionary is specified as "output_res" parameter
            ("output_res", None),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_actmap,
        )

    @e2e_pytest_unit
    def test_active_score_from_probs_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "active_score_from_probs" function input parameters validation

        <b>Input data:</b>
        "predictions" non-expected type object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "active_score_from_probs" function
        """
        with pytest.raises(ValueError):
            active_score_from_probs(predictions=None)  # type: ignore

    @e2e_pytest_unit
    def test_sigmoid_numpy_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "sigmoid_numpy" function input parameters validation

        <b>Input data:</b>
        "x" non-ndarray object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "sigmoid_numpy" function
        """
        with pytest.raises(ValueError):
            sigmoid_numpy(x="unexpected str")  # type: ignore

    @e2e_pytest_unit
    def test_softmax_numpy_parameters_params_validation(self):
        """
        <b>Description:</b>
        Check "softmax_numpy" function input parameters validation

        <b>Input data:</b>
        "x" non-ndarray object

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "softmax_numpy" function
        """
        with pytest.raises(ValueError):
            softmax_numpy(x="unexpected str")  # type: ignore

    @e2e_pytest_unit
    def test_get_multiclass_predictions_params_validation(self):
        """
        <b>Description:</b>
        Check "get_multiclass_predictions" function input parameters validation

        <b>Input data:</b>
        "get_multiclass_predictions" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_multiclass_predictions" function
        """
        label = LabelEntity(name="test label", domain=Domain.DETECTION)
        correct_values_dict = {
            "logits": np.random.randint(low=0, high=255, size=(10, 16, 3)),
            "labels": [label]
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "features" parameter
            ("logits", unexpected_str),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested label
            ("labels", [label, unexpected_str]),
            # Unexpected string is specified as "output_res" parameter
            ("activate", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_multiclass_predictions,
        )

    @e2e_pytest_unit
    def test_get_multilabel_predictions_params_validation(self):
        """
        <b>Description:</b>
        Check "get_multilabel_predictions" function input parameters validation

        <b>Input data:</b>
        "get_multilabel_predictions" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_multilabel_predictions" function
        """
        label = LabelEntity(name="test label", domain=Domain.DETECTION)
        correct_values_dict = {
            "logits": np.random.randint(low=0, high=255, size=(10, 16, 3)),
            "labels": [label]
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "features" parameter
            ("logits", unexpected_str),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested label
            ("labels", [label, unexpected_str]),
            # Unexpected string is specified as "pos_thr" parameter
            ("pos_thr", unexpected_str),
            # Unexpected string is specified as "output_res" parameter
            ("activate", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_multilabel_predictions,
        )

    @e2e_pytest_unit
    def test_get_hierarchical_predictions_params_validation(self):
        """
        <b>Description:</b>
        Check "get_hierarchical_predictions" function input parameters validation

        <b>Input data:</b>
        "get_hierarchical_predictions" unexpected type parameters

        <b>Expected results:</b>
        Test passes if ValueError exception is raised when unexpected type object is specified as
        input parameter for "get_hierarchical_predictions" function
        """
        label = LabelEntity(name="test label", domain=Domain.DETECTION)
        correct_values_dict = {
            "logits": np.random.randint(low=0, high=255, size=(10, 16, 3)),
            "labels": [label],
            "label_schema": LabelSchemaEntity(),
            "multihead_class_info": {"class": "info"},
        }
        unexpected_str = "unexpected string"
        unexpected_values = [
            # Unexpected string is specified as "features" parameter
            ("logits", unexpected_str),
            # Unexpected string is specified as "labels" parameter
            ("labels", unexpected_str),
            # Unexpected string is specified as nested label
            ("labels", [label, unexpected_str]),
            # Unexpected string is specified as "label_schema" parameter
            ("label_schema", unexpected_str),
            # Unexpected string is specified as "multihead_class_info" parameter
            ("multihead_class_info", unexpected_str),
            # Unexpected string is specified as "pos_thr" parameter
            ("pos_thr", unexpected_str),
            # Unexpected string is specified as "output_res" parameter
            ("activate", unexpected_str),
        ]

        check_value_error_exception_raised(
            correct_parameters=correct_values_dict,
            unexpected_values=unexpected_values,
            class_or_function=get_hierarchical_predictions,
        )
