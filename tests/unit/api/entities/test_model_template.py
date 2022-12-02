# Copyright (C) 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import copy
import itertools
from os import remove
from pathlib import Path

import pytest
import yaml

from otx.api.entities.label import Domain
from otx.api.entities.model_template import (
    ANOMALY_TASK_TYPES,
    TRAINABLE_TASK_TYPES,
    DatasetRequirements,
    Dependency,
    EntryPoints,
    ExportableCodePaths,
    HyperParameterData,
    InstantiationType,
    ModelOptimizationMethod,
    ModelTemplate,
    NullModelTemplate,
    TargetDevice,
    TaskFamily,
    TaskType,
    _parse_model_template_from_omegaconf,
    parse_model_template,
    parse_model_template_from_dict,
    task_type_to_label_domain,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


class CommonMethods:
    @staticmethod
    def cut_parameter_overrides_from_model_template(save_file: bool = True) -> dict:
        """Function saves config file with removed override_parameters node. Returns dictionary with path
        to new config file and dictionary with override_parameters"""
        with open(TestHyperParameterData().model_template_path()) as model_to_copy:
            model_data = yaml.safe_load(model_to_copy)
            override_parameters = model_data["hyper_parameters"]["parameter_overrides"]
        model_data["hyper_parameters"].pop("parameter_overrides")
        new_config_path = TestHyperParameterData.get_path_to_file(r"./no_overrides_template.yaml")
        if save_file:
            with open(new_config_path, "w+") as new_config:
                new_config.write(yaml.dump(model_data))
        return {
            "override_parameters": override_parameters,
            "new_config_path": new_config_path,
        }

    @staticmethod
    def check_model_attributes(model: ModelTemplate, expected_values: dict):
        assert model.model_template_id == expected_values.get("model_template_id")
        assert model.model_template_path == expected_values.get("model_template_path")
        assert model.name == expected_values.get("name")
        assert model.task_family == expected_values.get("task_family")
        assert model.task_type == expected_values.get("task_type")
        assert model.instantiation == expected_values.get("instantiation")
        assert model.summary == expected_values.get("summary", "")
        assert model.framework == expected_values.get("framework")
        assert model.max_nodes == expected_values.get("max_nodes", 1)
        assert model.application == expected_values.get("application")
        assert model.dependencies == expected_values.get("dependencies", [])
        assert model.initial_weights == expected_values.get("initial_weights")
        assert model.training_targets == expected_values.get("training_targets", [])
        assert model.inference_targets == expected_values.get("inference_targets", [])
        assert model.dataset_requirements == expected_values.get(
            "dataset_requirements", DatasetRequirements(classes=None)
        )
        assert model.model_optimization_methods == expected_values.get("model_optimization_methods", [])
        assert model.hyper_parameters == expected_values.get(
            "hyper_parameters",
            HyperParameterData(base_path=None, parameter_overrides={}),
        )
        assert model.is_trainable == expected_values.get("is_trainable", True)
        assert model.capabilities == expected_values.get("capabilities", [])
        assert model.grpc_address == expected_values.get("grpc_address")
        assert model.entrypoints == expected_values.get("entrypoints")
        assert model.exportable_code_paths == expected_values.get(
            "exportable_code_paths", ExportableCodePaths(default=None, openvino=None)
        )
        assert model.task_type_sort_priority == expected_values.get("task_type_sort_priority", -1)
        assert model.gigaflops == expected_values.get("gigaflops", 0)
        assert model.size == expected_values.get("size", 0)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTargetDevice:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_targetdevice(self):
        """
        <b>Description:</b>
        Check TargetDevice IntEnum class elements
        <b>Expected results:</b>
        Test passes if TargetDevice IntEnum class length is equal to expected value and its elements have expected
        sequence number
        """
        assert len(TargetDevice) == 4
        assert TargetDevice.UNSPECIFIED.value == 1
        assert TargetDevice.CPU.value == 2
        assert TargetDevice.GPU.value == 3
        assert TargetDevice.VPU.value == 4


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelOptimizationMethod:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_modeloptymizationmethod(self):
        """
        <b>Description:</b>
        Check ModelOptimizationMethod Enum class elements
        <b>Expected results:</b>
        Test passes if ModelOptimizationMethod Enum class length, methods and attributes return expected values
        <b>Steps</b>
        1. Check ModelOptimizationMethod length
        2. Check ModelOptimizationMethod elements value attribute
        3. Check ModelOptimizationMethod str method
        """
        assert len(ModelOptimizationMethod) == 2
        assert ModelOptimizationMethod.TENSORRT.value == 1
        assert ModelOptimizationMethod.OPENVINO.value == 2
        assert str(ModelOptimizationMethod.TENSORRT) == "TENSORRT"
        assert str(ModelOptimizationMethod.OPENVINO) == "OPENVINO"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDatasetRequirements:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_datasetrequirements(self):
        """
        <b>Description:</b>
        Check DatasetRequirements dataclass
        <b>Expected results:</b>
        Test passes if classes attribute of DatasetRequirements dataclass returns expected values
        """
        classes_list = ["class_1", "class_2"]
        test_dataset_requirements = DatasetRequirements(classes_list)
        equal_dataset_requirements = DatasetRequirements(classes_list)
        other_test_dataset_requirements = DatasetRequirements(["class_1", "class_3"])
        assert test_dataset_requirements.classes == ["class_1", "class_2"]
        assert other_test_dataset_requirements.classes == ["class_1", "class_3"]
        assert test_dataset_requirements == equal_dataset_requirements
        assert test_dataset_requirements != other_test_dataset_requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestExportableCodePaths:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_exportablecodepaths(self):
        """
        <b>Description:</b>
        Check ExportableCodePaths dataclass
        <b>Expected results:</b>
        Test passes if default, openvino attributes of ExportableCodePaths dataclass return expected values
        """
        exportable_code_paths = ExportableCodePaths("default code path", "openvino code path")
        equal_exportable_code_paths = ExportableCodePaths("default code path", "openvino code path")
        unequal_exportable_code_paths = ExportableCodePaths("other default code path", "openvino code path")
        assert exportable_code_paths.default == "default code path"
        assert exportable_code_paths.openvino == "openvino code path"
        assert exportable_code_paths == equal_exportable_code_paths
        assert exportable_code_paths != unequal_exportable_code_paths


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTaskFamily:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_taskfamily(self):
        """
        <b>Description:</b>
        Check TaskFamily Enum class elements
        <b>Expected results:</b>
        Test passes if TaskFamily Enum class length, attributes and methods return expected values
        <b>Steps</b>
        1. Check TaskFamily length
        2. Check TaskFamily elements value attribute
        3. Check TaskFamily str method
        """
        assert len(TaskFamily) == 3
        assert TaskFamily.VISION.value == 1
        assert TaskFamily.FLOW_CONTROL.value == 2
        assert TaskFamily.DATASET.value == 3
        assert str(TaskFamily.VISION) == "VISION"
        assert str(TaskFamily.FLOW_CONTROL) == "FLOW_CONTROL"
        assert str(TaskFamily.DATASET) == "DATASET"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTaskType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_tasktype(self):
        """
        <b>Description:</b>
        Check TaskType Enum class elements
        <b>Expected results:</b>
        Test passes if TaskType Enum class length, attributes and methods return expected values
        <b>Steps</b>
        1. Check TaskType length
        2. Check TaskType elements value attribute
        3. Check TaskType str method
        """
        assert len(TaskType) == 15
        assert TaskType.NULL.value == 1
        assert TaskType.DATASET.value == 2
        assert TaskType.CLASSIFICATION.value == 3
        assert TaskType.SEGMENTATION.value == 4
        assert TaskType.DETECTION.value == 5
        assert TaskType.ANOMALY_DETECTION.value == 6
        assert TaskType.CROP.value == 7
        assert TaskType.TILE.value == 8
        assert TaskType.INSTANCE_SEGMENTATION.value == 9
        assert TaskType.ACTIVELEARNING.value == 10
        assert TaskType.ANOMALY_SEGMENTATION.value == 11
        assert TaskType.ANOMALY_CLASSIFICATION.value == 12
        assert TaskType.ROTATED_DETECTION.value == 13
        assert TaskType.ACTION_CLASSIFICATION.value == 14
        assert TaskType.ACTION_DETECTION.value == 15
        assert str(TaskType.NULL) == "NULL"
        assert str(TaskType.DATASET) == "DATASET"
        assert str(TaskType.CLASSIFICATION) == "CLASSIFICATION"
        assert str(TaskType.SEGMENTATION) == "SEGMENTATION"
        assert str(TaskType.DETECTION) == "DETECTION"
        assert str(TaskType.ANOMALY_DETECTION) == "ANOMALY_DETECTION"
        assert str(TaskType.CROP) == "CROP"
        assert str(TaskType.TILE) == "TILE"
        assert str(TaskType.INSTANCE_SEGMENTATION) == "INSTANCE_SEGMENTATION"
        assert str(TaskType.ACTIVELEARNING) == "ACTIVELEARNING"
        assert str(TaskType.ANOMALY_SEGMENTATION) == "ANOMALY_SEGMENTATION"
        assert str(TaskType.ANOMALY_CLASSIFICATION) == "ANOMALY_CLASSIFICATION"
        assert str(TaskType.ROTATED_DETECTION) == "ROTATED_DETECTION"
        assert str(TaskType.ACTION_CLASSIFICATION) == "ACTION_CLASSIFICATION"
        assert str(TaskType.ACTION_DETECTION) == "ACTION_DETECTION"

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_tasktype_to_label_domain(self):
        """
        <b>Description:</b>
        Check task_type_to_label_domain function
        <b>Expected results:</b>
        Test passes if task_type_to_label_domain function returns expected mapping of TaskType class element
        to Domain class element
        expected values
        <b>Steps</b>
        1. Check positive scenario with existing TaskType element
        2. Check ValueError exception raised when requests non-existing TaskType element
        """
        assert task_type_to_label_domain(TaskType.CLASSIFICATION) == Domain.CLASSIFICATION
        assert task_type_to_label_domain(TaskType.DETECTION) == Domain.DETECTION
        assert task_type_to_label_domain(TaskType.SEGMENTATION) == Domain.SEGMENTATION
        assert task_type_to_label_domain(TaskType.INSTANCE_SEGMENTATION) == Domain.INSTANCE_SEGMENTATION
        assert task_type_to_label_domain(TaskType.ANOMALY_CLASSIFICATION) == Domain.ANOMALY_CLASSIFICATION
        assert task_type_to_label_domain(TaskType.ANOMALY_DETECTION) == Domain.ANOMALY_DETECTION
        assert task_type_to_label_domain(TaskType.ANOMALY_SEGMENTATION) == Domain.ANOMALY_SEGMENTATION
        for not_mapped_task in [
            TaskType.NULL,
            TaskType.DATASET,
            TaskType.CROP,
            TaskType.TILE,
            TaskType.ACTIVELEARNING,
            "key",
        ]:
            with pytest.raises(ValueError):
                task_type_to_label_domain(not_mapped_task)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestHyperParameterData:
    @staticmethod
    def get_path_to_file(filename: str) -> str:
        """Return the path to the file named 'filename', which saved in tests/entities directory"""
        return str(Path(__file__).parent / Path(filename))

    def config_path(self) -> str:
        return self.get_path_to_file(r"./dummy_config.yaml")

    def model_template_path(self) -> str:
        return self.get_path_to_file(r"./dummy_template.yaml")

    @staticmethod
    def parameter_overrides() -> dict:
        return {
            "learning_parameters": {
                "batch_size": {"default_value": 10},
                "learning_rate": {"default_value": 0.5},
                "learning_rate_warmup_iters": {"min_value": 10},
                "num_checkpoints": {"max_value": 95},
                "num_iters": {"min_value": 2},
                "num_workers": {"description": "New workers description"},
            },
            "postprocessing": {
                "confidence_threshold": {"default_value": 0.4},
                "result_based_confidence_threshold": {"header": "New header"},
            },
        }

    def remove_value_key_from_config(self, config_content: dict) -> None:
        """Function removes "value" key from config dictionary"""
        config_content_copy = copy.deepcopy(config_content)
        for key, value in config_content_copy.items():
            if isinstance(value, dict):
                if key != "ui_rules":
                    self.remove_value_key_from_config(config_content[key])
            elif key == "value":
                config_content.pop(key)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_hyperparameterdata_data(self):
        """
        <b>Description:</b>
        Check data property of HyperParameterData class instance
        <b>Expected results:</b>
        Test passes if data property of HyperParameterData class instance has expected values
        <b>Steps</b>
        1. Check that data attribute of HyperParameterData object has empty dictionary before using load_parameters
        method
        2. Check scenario when path to config.yaml file specified in base_path parameter during HyperParameterData
        object initiation
        """

        def open_config_file_and_remove_value_key() -> dict:
            """Function returns config file dictionary with removed value key"""
            with open(TestHyperParameterData().config_path()) as config_file:
                config_content = yaml.safe_load(config_file)
                self.remove_value_key_from_config(config_content)
                return config_content

        # Forming expected data dictionary
        expected_config_data = open_config_file_and_remove_value_key()
        # Creating model template with no override_params_node
        model_template = CommonMethods.cut_parameter_overrides_from_model_template()
        model_template_path = model_template.get("new_config_path")
        # Checking data attribute of HyperParameterData class instance
        hyper_parameter_data = HyperParameterData(base_path=self.config_path())
        assert hyper_parameter_data.data == {}
        hyper_parameter_data.load_parameters(model_template_path)
        assert hyper_parameter_data.data == expected_config_data
        remove(model_template_path)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_remove_parameter_values_from_data(self):
        """
        <b>Description:</b>
        Check remove_parameter_values_from_data method of HyperParameterData class instance
        <b>Expected results:</b>
        Test passes if data property of HyperParameterData class instance has no "value" key after using
        load_parameters method
        <b>Steps</b>
        1. Check that yaml config file has "value" keys
        2. Check that data property of HyperParameterData class instance has no value key after using
        load_parameters method
        """

        def search_value_key_in_config(config_content: dict) -> bool:
            """
            Function returns "True" if "value" key presents in config dictionary and "False" if absents
            """
            is_value_key_exists = False
            config_content_copy = copy.deepcopy(config_content)
            for key, value in config_content_copy.items():
                if isinstance(value, dict):
                    if key != "ui_rules":
                        if search_value_key_in_config(config_content[key]):
                            is_value_key_exists = True
                            break
                elif key == "value":
                    is_value_key_exists = True
            return is_value_key_exists

        with open(self.config_path()) as config_file:
            config_data = yaml.safe_load(config_file)
        # Checking test dataset
        assert search_value_key_in_config(config_data)
        model_template = CommonMethods.cut_parameter_overrides_from_model_template()
        model_template_path = model_template.get("new_config_path")
        hyper_parameter_data = HyperParameterData(base_path=self.config_path())
        hyper_parameter_data.load_parameters(model_template_path)
        assert not search_value_key_in_config(hyper_parameter_data.data)
        remove(model_template_path)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_hyperparameterdata_has_overrides(self):
        """
        <b>Description:</b>
        Check has_overrides property of HyperParameterData class instance
        <b>Expected results:</b>
        Test passes if has_overrides property of HyperParameterData class instance has expected values
        <b>Steps</b>
        1. Check scenario when HyperParameterData object has specified parameter_overrides dictionary
        2. Check scenario when parameter_overrides dictionary not specified for HyperParameterData object
        """
        model_template = CommonMethods.cut_parameter_overrides_from_model_template()
        model_template_path = model_template.get("new_config_path")
        hyper_parameter_data = HyperParameterData(
            base_path=self.config_path(),
            parameter_overrides=model_template.get("override_parameters"),
        )
        hyper_parameter_data.load_parameters(model_template_path)
        has_overrides_hyper_parameter_data = HyperParameterData(
            base_path=self.config_path(), parameter_overrides=self.parameter_overrides()
        )
        assert has_overrides_hyper_parameter_data.has_overrides
        has_overrides_hyper_parameter_data.load_parameters(self.model_template_path())
        assert has_overrides_hyper_parameter_data.has_overrides
        no_overrides_hyper_parameter_data = HyperParameterData(base_path=self.config_path())
        assert not no_overrides_hyper_parameter_data.has_overrides
        no_overrides_hyper_parameter_data.load_parameters(model_template_path)
        assert not no_overrides_hyper_parameter_data.has_overrides
        remove(model_template_path)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_hyperparameterdata_has_valid_configurable_parameters(self):
        """
        <b>Description:</b>
        Check has_valid_configurable_parameters property of HyperParameterData class instance
        <b>Expected results:</b>
        Test passes if has_valid_configurable_parameters property of HyperParameterData class instance has
        expected values
        <b>Steps</b>
        1. Check scenario when base_path parameter specified during HyperParameterData object initiation and
        model_template_path file specified in load_parameters method exists
        2. Check scenario when __has_valid_configurable_parameters returns False when base_path parameter
        not specified during HyperParameterData object initiation
        3. Check scenario when __has_valid_configurable_parameters returns False when model_template_path file
        specified in load_parameters method not exists
        4. Check that ValueError exception raised when base_path file has unexpected structure
        """
        # positive scenario when expected has_valid_configurable_parameters = True
        model_template = CommonMethods.cut_parameter_overrides_from_model_template()
        model_template_path = model_template.get("new_config_path")
        hyper_parameter_data = HyperParameterData(base_path=self.config_path())
        hyper_parameter_data.load_parameters(model_template_path)
        assert hyper_parameter_data.has_valid_configurable_parameters
        # scenario when base_path parameter not specified
        no_config_specified_data = HyperParameterData()
        no_config_specified_data.load_parameters(model_template_path)
        assert not no_config_specified_data.base_path
        assert not no_config_specified_data.has_valid_configurable_parameters
        # scenario when model_template_path parameter not exists
        config_path = self.config_path()
        no_model_specified_data = HyperParameterData(config_path)
        no_model_specified_data.load_parameters(r"./file_not_exists.yaml")
        assert no_model_specified_data.base_path == config_path
        assert not no_config_specified_data.has_valid_configurable_parameters
        # check for incorrect config file
        incorrect_config_yaml_path = self.get_path_to_file(r"./incorrect_config.yaml")
        with open(incorrect_config_yaml_path, "w+") as incorrect_yaml_file:
            incorrect_yaml_file.write("[]")
        incorrect_config_data = HyperParameterData(incorrect_config_yaml_path)
        with pytest.raises(ValueError):
            incorrect_config_data.load_parameters(model_template_path)
        remove(model_template_path)
        remove(incorrect_config_yaml_path)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_hyperparameterdata_substitute_parameter_overrides(self):
        """
        <b>Description:</b>
        Check substitute_parameter_overrides method of HyperParameterData class instance
        <b>Expected results:</b>
        Test passes if substitute_parameter_overrides method returns data attribute of HyperParameterData class
        instance has expected value
        <b>Steps</b>
        1. Check positive scenario with valid parameter_overrides dictionary
        2. Check negative scenario with unexpected key to override
        3. Check negative scenario with "value" key to override in parameter_overrides dictionary
        """
        # positive scenario with valid overrides parameters dictionary
        model_template = CommonMethods.cut_parameter_overrides_from_model_template()
        model_template_path = model_template.get("new_config_path")
        hyper_parameter_data = HyperParameterData(
            base_path=self.config_path(), parameter_overrides=self.parameter_overrides()
        )
        hyper_parameter_data.load_parameters(model_template_path)
        learning_parameters_data = hyper_parameter_data.data.get("learning_parameters")
        postprocessing_data = hyper_parameter_data.data.get("postprocessing")
        assert learning_parameters_data.get("batch_size").get("default_value") == 10
        assert learning_parameters_data.get("learning_rate").get("default_value") == 0.5
        assert learning_parameters_data.get("learning_rate_warmup_iters").get("min_value") == 10
        assert learning_parameters_data.get("num_checkpoints").get("max_value") == 95
        assert learning_parameters_data.get("num_iters").get("min_value") == 2
        assert learning_parameters_data.get("num_workers").get("description") == "New workers description"
        assert postprocessing_data.get("confidence_threshold").get("default_value") == 0.4
        assert postprocessing_data.get("result_based_confidence_threshold").get("header") == "New header"
        # negative scenario with key not specified in config.yaml file
        unexpected_key_dict = {
            "learning_parameters": {"batch_size": {"default_value": 10}},
            "unexpected_key": {"parameter1": 1},
        }
        hyper_parameter_data = HyperParameterData(base_path=self.config_path(), parameter_overrides=unexpected_key_dict)
        with pytest.raises(ValueError):
            hyper_parameter_data.load_parameters(self.model_template_path())
        # negative scenario with "value" key not allowed to override
        restricted_key_dict = {"learning_parameters": {"batch_size": {"default_value": 10, "value": 1}}}
        hyper_parameter_data = HyperParameterData(base_path=self.config_path(), parameter_overrides=restricted_key_dict)
        with pytest.raises(KeyError):
            hyper_parameter_data.load_parameters(self.model_template_path())

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_manually_set_data_and_validate(self):
        """
        <b>Description:</b>
        Check manually_set_data_and_validate method of HyperParameterData class instance
        <b>Expected results:</b>
        Test passes if manually_set_data_and_validate method of HyperParameterData class instance has expected values
        <b>Steps</b>
        1. Check scenario when manually_set_data_and_validate method overrides HyperParameterData class data
        2. Check scenario when manually_set_data_and_validate method sets HyperParameterData class data
        3. Check scenario when manually_set_data_and_validate method removes HyperParameterData class data
        """
        # Check for override class data
        model_template = CommonMethods.cut_parameter_overrides_from_model_template()
        model_template_path = model_template.get("new_config_path")
        hyper_parameter_data = HyperParameterData(base_path=self.config_path())
        hyper_parameter_data.load_parameters(model_template_path)
        parameter_overrides = self.parameter_overrides()
        hyper_parameter_data.manually_set_data_and_validate(parameter_overrides)
        assert hyper_parameter_data.data == parameter_overrides
        assert hyper_parameter_data.has_valid_configurable_parameters
        # Check for set class data
        with open(self.config_path()) as config_file:
            data_to_set = yaml.safe_load(config_file)
        set_hyper_parameter_data = HyperParameterData(self.config_path())
        set_hyper_parameter_data.manually_set_data_and_validate(data_to_set)
        assert set_hyper_parameter_data.data == data_to_set
        assert set_hyper_parameter_data.has_valid_configurable_parameters
        # Check for set empty class data
        empty_hyper_parameter_data = HyperParameterData(self.config_path())
        empty_hyper_parameter_data.manually_set_data_and_validate({})
        assert empty_hyper_parameter_data.data == {}
        assert empty_hyper_parameter_data.has_valid_configurable_parameters
        remove(model_template_path)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestInstantiationType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_instantiationtype(self):
        """
        <b>Description:</b>
        Check InstantiationType Enum class elements
        <b>Expected results:</b>
        Test passes if InstantiationType Enum class length, methods and attributes return expected values
        <b>Steps</b>
        1. Check InstantiationType length
        2. Check InstantiationType elements value attribute
        3. Check InstantiationType str method
        """
        assert len(InstantiationType) == 3
        assert InstantiationType.NONE.value == 1
        assert InstantiationType.CLASS.value == 2
        assert InstantiationType.GRPC.value == 3
        assert str(InstantiationType.NONE) == "NONE"
        assert str(InstantiationType.CLASS) == "CLASS"
        assert str(InstantiationType.GRPC) == "GRPC"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestDependency:
    @staticmethod
    def dependency_parameters() -> dict:
        return {
            "source": "dependency source",
            "destination": "dependency destination",
            "size": 1024,
            "sha256": "wgyjp0xks3obuwiu0jqea3q94ninfo0dgphv2t57wv1tq6qlwtr3jitvn0uo8a14",
        }

    def dependency(self) -> Dependency:
        return Dependency(**self.dependency_parameters())

    @staticmethod
    def unequal_dependency_parameters() -> dict:
        return {
            "source": "other source",
            "destination": "other destination",
            "size": 512,
            "sha256": "q0y3vrh9wcff5lc77epfx08pb6ioredv341u68rp1qvxyl41wzt9tlih94s5273i",
        }

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_depencdency(self):
        """
        <b>Description:</b>
        Check Dependency dataclass elements
        <b>Expected results:</b>
        Test passes if attributes of Dependency dataclass return expected values
        <b>Steps</b>
        1. Check Dependency source, destination, size and sha256 attributes of initiated Dependency object
        2. Check comparing Dependency object with equal object
        3. Check comparing Dependency object with unequal object
        """
        # Checking Dependency object attributes
        expected_source = self.dependency_parameters().get("source")
        expected_destination = self.dependency_parameters().get("destination")
        dependency = self.dependency()
        assert dependency.source == expected_source
        assert dependency.destination == expected_destination
        assert dependency.size == self.dependency_parameters().get("size")
        assert dependency.sha256 == self.dependency_parameters().get("sha256")
        default_attributes_dependency = Dependency(source=expected_source, destination=expected_destination)
        assert default_attributes_dependency.source == expected_source
        assert default_attributes_dependency.destination == expected_destination
        assert not default_attributes_dependency.size
        assert not default_attributes_dependency.sha256
        # Comparing Dependency object with equal
        equal_dependency = self.dependency()
        assert dependency == equal_dependency
        # Comparing Dependency object with unequal
        keys_list = ["source", "destination", "size", "sha256"]
        parameter_combinations = []
        for i in range(1, len(keys_list) + 1):
            parameter_combinations.append(list(itertools.combinations(keys_list, i)))
        # In each of scenario creating a copy of equal parameters and replacing to values from prepared
        # dictionary
        unequal_dependency_parameters = self.unequal_dependency_parameters()
        for scenario in parameter_combinations:
            for parameters in scenario:
                unequal_params_dict = dict(self.dependency_parameters())
                for key in parameters:
                    unequal_params_dict[key] = unequal_dependency_parameters.get(key)
                unequal_dependency = Dependency(**unequal_params_dict)
                assert dependency != unequal_dependency, (
                    "Failed to check that Dependency instances with different " f"{parameters} are unequal"
                )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestEntryPoints:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_entrypoints(self):
        """
        <b>Description:</b>
        Check EntryPoints dataclass elements
        <b>Expected results:</b>
        Test passes if classes attributes of EntryPoints dataclass return expected values
        <b>Steps</b>
        1. Check EntryPoints base, openvino and nncf attributes of initiated Dependency object
        2. Check comparing EntryPoints object with equal object
        3. Check comparing EntryPoints object with unequal object
        """
        base = "base interface"
        openvino = "OpenVINO interface"
        nncf = "NNCF interface"
        entrypoints_parameters = {"base": base, "openvino": openvino, "nncf": nncf}
        # Checking EntryPoints object attributes
        entry_points = EntryPoints(**entrypoints_parameters)
        assert entry_points.base == base
        assert entry_points.openvino == openvino
        assert entry_points.nncf == nncf
        default_attributes_entrypoints = EntryPoints(base=base)
        assert default_attributes_entrypoints.base == base
        assert not default_attributes_entrypoints.openvino
        assert not default_attributes_entrypoints.nncf
        # Comparing EntryPoints object with equal
        equal_entry_points = EntryPoints(**entrypoints_parameters)
        assert entry_points == equal_entry_points
        # Comparing Dependency object with unequal
        keys_list = ["base", "openvino", "nncf"]
        parameter_combinations = []
        for i in range(1, len(keys_list) + 1):
            parameter_combinations.append(list(itertools.combinations(keys_list, i)))
        # In each of scenario creating a copy of equal parameters and replacing to values from prepared
        # dictionary
        unequal_entrypoints_parameters = {
            "base": "other base interface",
            "openvino": "other OpenVINO interface",
            "nncf": "other NNCF interface",
        }
        for scenario in parameter_combinations:
            for parameters in scenario:
                unequal_params_dict = dict(entrypoints_parameters)
                for key in parameters:
                    unequal_params_dict[key] = unequal_entrypoints_parameters.get(key)
                unequal_entry_points = EntryPoints(**unequal_params_dict)
                assert entry_points != unequal_entry_points, (
                    "Failed to check that EntryPoints instances with " f"different {parameters} are unequal"
                )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelTemplate:
    @staticmethod
    def default_model_parameters() -> dict:
        return {
            "model_template_id": "A16",
            "model_template_path": TestHyperParameterData().model_template_path(),
            "name": "test_model",
            "task_family": TaskFamily.DATASET,
            "task_type": TaskType.DETECTION,
            "instantiation": InstantiationType.NONE,
        }

    def optional_model_parameters(self):
        model_template = CommonMethods.cut_parameter_overrides_from_model_template()
        model_template_path = model_template.get("new_config_path")
        hyper_parameter_data = HyperParameterData(base_path=TestHyperParameterData().config_path())
        hyper_parameter_data.load_parameters(model_template_path)
        optional_parameters = dict(self.default_model_parameters())
        optional_parameters["model_template_path"] = model_template_path
        optional_parameters["model_template_id"] = "B18"
        optional_parameters["name"] = "test_model2"
        optional_parameters["task_family"] = TaskFamily.VISION
        optional_parameters["summary"] = "algorithm related information"
        optional_parameters["framework"] = "test framework"
        optional_parameters["max_nodes"] = 2
        optional_parameters["application"] = "test application"
        optional_parameters["dependencies"] = [
            TestDependency().dependency(),
            Dependency(**TestDependency().unequal_dependency_parameters()),
        ]
        optional_parameters["initial_weights"] = "https://some_url.com"
        optional_parameters["training_targets"] = [
            TargetDevice.CPU,
            TargetDevice.GPU,
            TargetDevice.VPU,
        ]
        optional_parameters["inference_targets"] = [
            TargetDevice.CPU,
            TargetDevice.VPU,
            TargetDevice.GPU,
        ]
        optional_parameters["dataset_requirements"] = [
            DatasetRequirements(["class1", "class_2"]),
            DatasetRequirements(["class_3", "class_4"]),
        ]
        optional_parameters["model_optimization_methods"] = [
            ModelOptimizationMethod.OPENVINO,
            ModelOptimizationMethod.TENSORRT,
        ]
        optional_parameters["hyper_parameters"] = hyper_parameter_data
        optional_parameters["is_trainable"] = False
        optional_parameters["capabilities"] = [
            "compute_uncertainty_score",
            "compute_representations",
        ]
        optional_parameters["grpc_address"] = "192.168.1.1"
        optional_parameters["entrypoints"] = EntryPoints.openvino
        optional_parameters["exportable_code_paths"] = ExportableCodePaths.openvino
        optional_parameters["task_type_sort_priority"] = 0
        optional_parameters["gigaflops"] = 1
        optional_parameters["size"] = 1024
        return optional_parameters

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_template_initiation(self):
        """
        <b>Description:</b>
        Check TestModelTemplate dataclass attributes
        <b>Expected results:</b>
        Test passes if classes attributes of ModelTemplate object return expected values
        <b>Steps</b>
        1. Check attributes of ModelTemplate object initiated with default values
        2. Check attributes of ModelTemplate object initiated with fully specified values
        """
        # Checks for object with default values
        default_model_template = ModelTemplate(**self.default_model_parameters())
        CommonMethods.check_model_attributes(default_model_template, self.default_model_parameters())
        # Checks for object with specified values
        optional_parameters_model = ModelTemplate(**self.optional_model_parameters())
        CommonMethods.check_model_attributes(optional_parameters_model, self.optional_model_parameters())
        assert default_model_template != optional_parameters_model
        remove(optional_parameters_model.model_template_path)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_template_post_initialization(self):
        """
        <b>Description:</b>
        Check __post_init__ method of ModelTemplate dataclass
        <b>Expected results:</b>
        Test passes if __post_init__ method of ModelTemplate object raises ValueError exception for parameters violation
        <b>Steps</b>
        1. Check ValueError exception when initialized ModelTemplate has gRPC instantiation and "" gRPC address
        2. Check ValueError exception when initialized ModelTemplate has CLASS instantiation and no entry points
        3. Check ValueError exception when initialized ModelTemplate has VISION task family and has no specified path to
        config file
        4. Check ValueError exception when initialized ModelTemplate with task family in not equal to VISION and but has
        specified path to config file
        """
        model_template_parameters = self.default_model_parameters()
        # gRPC instantiation and "" GRPC address
        grpc_address_empty = dict(model_template_parameters)
        grpc_address_empty["instantiation"] = InstantiationType.GRPC
        grpc_address_empty["grpc_address"] = ""
        with pytest.raises(ValueError):
            ModelTemplate(**grpc_address_empty)
        # CLASS instantiation and no entrypoints
        class_no_entrypoints = dict(model_template_parameters)
        class_no_entrypoints["instantiation"] = InstantiationType.CLASS
        with pytest.raises(ValueError):
            ModelTemplate(**class_no_entrypoints)
        # VISION task family no config file path specified
        vision_no_config = dict(model_template_parameters)
        vision_no_config["task_family"] = TaskFamily.VISION
        with pytest.raises(ValueError):
            ModelTemplate(**vision_no_config)
        # Not VISION task family and specified path to config
        class_not_vision_with_config = dict(model_template_parameters)
        class_not_vision_with_config["hyper_parameters"] = HyperParameterData(
            base_path=TestHyperParameterData().config_path()
        )
        with pytest.raises(ValueError):
            ModelTemplate(**class_not_vision_with_config)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_template_capabilities(self):
        """
        <b>Description:</b>
        Check computes_uncertainty_score and computes_representations methods of ModelTemplate dataclass
        <b>Expected results:</b>
        Test passes if computes_uncertainty_score and computes_representations methods of ModelTemplate object return
        expected bool values related to capabilities attribute
        <b>Steps</b>
        1. Check computes_uncertainty_score and computes_representations methods return True value for ModelTemplate
        2. Check computes_uncertainty_score method returns True and computes_representations returns False value
        3. Check computes_uncertainty_score method returns False and computes_representations returns True value
        4. Check computes_uncertainty_score and computes_representations methods return False value
        """
        # Check for computes_uncertainty_score and computes_representations methods return True
        score_representations_model = ModelTemplate(**self.optional_model_parameters())
        assert score_representations_model.computes_uncertainty_score()
        assert score_representations_model.computes_representations()
        model_template_parameters = self.default_model_parameters()
        # Check for computes_uncertainty_score is True and computes_representations is False
        score_true_presentations_false_parameters = dict(model_template_parameters)
        score_true_presentations_false_parameters["capabilities"] = [
            "compute_uncertainty_score",
            "not test parameter",
        ]
        score_true_presentations_false_model = ModelTemplate(**score_true_presentations_false_parameters)
        assert score_true_presentations_false_model.computes_uncertainty_score()
        assert not score_true_presentations_false_model.computes_representations()
        # Check for computes_uncertainty_score is False and computes_representations is True
        score_true_presentations_false_parameters = dict(model_template_parameters)
        score_true_presentations_false_parameters["capabilities"] = [
            "compute_representations",
            "not test parameter",
        ]
        score_true_presentations_false_model = ModelTemplate(**score_true_presentations_false_parameters)
        assert not score_true_presentations_false_model.computes_uncertainty_score()
        assert score_true_presentations_false_model.computes_representations()
        # Check for computes_uncertainty_score and computes_representations methods return False
        no_score_representations_model = ModelTemplate(**self.default_model_parameters())
        assert not no_score_representations_model.computes_uncertainty_score()
        assert not no_score_representations_model.computes_representations()
        remove(score_representations_model.model_template_path)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_template_is_task_global(self):
        """
        <b>Description:</b>
        Check is_task_global method of ModelTemplate dataclass
        <b>Expected results:</b>
        Test passes if is_task_global method of ModelTemplate object returns expected bool values related to
        task_type attribute
        <b>Steps</b>
        1. Check is_task_global method returns True if task_type equal to CLASSIFICATION or ANOMALY_CLASSIFICATION
        2. Check is_task_global method returns False if task_type not equal to CLASSIFICATION or ANOMALY_CLASSIFICATION
        """
        # Check is_task_global method returns True for CLASSIFICATION and ANOMALY_CLASSIFICATION
        for global_task_type in (
            TaskType.CLASSIFICATION,
            TaskType.ANOMALY_CLASSIFICATION,
        ):
            default_parameters = self.default_model_parameters()
            task_global_parameters = dict(default_parameters)
            task_global_parameters["task_type"] = global_task_type
            task_global_model_template = ModelTemplate(**task_global_parameters)
            assert (
                task_global_model_template.is_task_global()
            ), f"Expected True value returned by is_task_global for {global_task_type}"
        # Check is_task_global method returns False for the other tasks
        non_global_task_parameters = dict(default_parameters)
        non_global_tasks_list = []
        for task_type in TaskType:
            if not task_type.is_global:
                non_global_tasks_list.append(task_type)
        for non_global_task in non_global_tasks_list:
            non_global_task_parameters["task_type"] = non_global_task
            non_global_task_template = ModelTemplate(**non_global_task_parameters)
            assert not non_global_task_template.is_task_global(), (
                f"Expected False value returned by is_task_global method for {non_global_task}, "
                f"only CLASSIFICATION and ANOMALY_CLASSIFICATION task types are global"
            )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestNullModelTemplate:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_null_mode_attributes(self):
        """
        <b>Description:</b>
        Check attributes of NullModelTemplate class object
        <b>Expected results:</b>
        Test passes if NullModelTemplate class object attributes have expected values
        """
        null_model_template = NullModelTemplate()
        expected_null_model_parameters = {
            "model_template_id": "",
            "model_template_path": "",
            "name": "Null algorithm",
            "task_family": TaskFamily.FLOW_CONTROL,
            "task_type": TaskType.NULL,
            "instantiation": InstantiationType.NONE,
        }
        CommonMethods.check_model_attributes(null_model_template, expected_null_model_parameters)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTaskTypesConstants:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_task_type_constants(self):
        """
        <b>Description:</b>
        Check values of ANOMALY_TASK_TYPES and TRAINABLE_TASK_TYPES constants
        <b>Expected results:</b>
        Test passes if ANOMALY_TASK_TYPES and TRAINABLE_TASK_TYPES constants return expected values
        """
        assert ANOMALY_TASK_TYPES == (
            TaskType.ANOMALY_DETECTION,
            TaskType.ANOMALY_CLASSIFICATION,
            TaskType.ANOMALY_SEGMENTATION,
        )
        assert TRAINABLE_TASK_TYPES == (
            TaskType.CLASSIFICATION,
            TaskType.DETECTION,
            TaskType.SEGMENTATION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.ANOMALY_DETECTION,
            TaskType.ANOMALY_CLASSIFICATION,
            TaskType.ANOMALY_SEGMENTATION,
            TaskType.ROTATED_DETECTION,
            TaskType.ACTION_CLASSIFICATION,
            TaskType.ACTION_DETECTION,
        )


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestParseModelTemplate:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_parse_model_template_from_omegaconf(self):
        """
        <b>Description:</b>
        Check _parse_model_template_from_omegaconf function returns expected instance of ModelTemplate class
        <b>Expected results:</b>
        Test passes if _parse_model_template_from_omegaconf function returns expected instance of ModelTemplate
        class
        """
        model_template_path = TestHyperParameterData().model_template_path()
        with open(model_template_path) as model_template_file:
            model_template_content = yaml.safe_load(model_template_file)
            model_template_content["model_template_id"] = model_template_content["name"].replace(" ", "_")
            model_template_content["model_template_path"] = model_template_path
        parsed_model_template = _parse_model_template_from_omegaconf(model_template_content)
        assert isinstance(parsed_model_template, ModelTemplate)
        # Forming expected hyper parameter data dictionary from config.yaml and overridden parameters from
        # model_template,yaml
        parameter_overrides = {
            "learning_parameters": {
                "batch_size": {"default_value": 64},
                "learning_rate": {"default_value": 0.05},
                "learning_rate_warmup_iters": {"default_value": 100},
                "num_iters": {"default_value": 13000},
            }
        }
        expected_hyper_parameters = HyperParameterData(
            base_path="./dummy_config.yaml", parameter_overrides=parameter_overrides
        )
        expected_hyper_parameters.load_parameters(model_template_path)
        expected_parsed_model_parameters = {
            "entrypoints": EntryPoints(base="base entrypoints", openvino=None, nncf=None),
            "framework": "Test framework",
            "hyper_parameters": expected_hyper_parameters,
            "inference_targets": [
                TargetDevice.CPU,
                TargetDevice.GPU,
                TargetDevice.VPU,
            ],
            "instantiation": InstantiationType.CLASS,
            "model_template_id": "Custom_Object_Detection_--_TEST_ONLY",
            "model_template_path": model_template_path,
            "name": "Custom Object Detection -- TEST ONLY",
            "summary": "Fast and lightweight object detector.",
            "task_family": TaskFamily.VISION,
            "task_type": TaskType.DETECTION,
            "training_targets": [TargetDevice.GPU, TargetDevice.CPU],
        }
        CommonMethods.check_model_attributes(parsed_model_template, expected_parsed_model_parameters)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_parse_model_template(self):
        """
        <b>Description:</b>
        Check parse_model_template function returns expected instance of ModelTemplate class
        <b>Expected results:</b>
        Test passes if parse_model_template function returns instance of ModelTemplate class
        <b>Steps</b>
        1. Check model_template_id and model_template_path attributes of ModelTemplate instance returned by
        parse_model_template function for template file with not specified model_template_id parameter
        2. Check model_template_id and model_template_path attributes of ModelTemplate instance returned by
        parse_model_template function for template file with specified model_template_id parameter
        3. Check ValueError exception raised if path to list-type template file is specified as input parameter in
        parse_model_template function
        """
        # Check for template file with not specified model_template_id
        model_template_path = TestHyperParameterData().model_template_path()
        not_specified_id_template = parse_model_template(model_template_path)
        assert not_specified_id_template.model_template_id == "Custom_Object_Detection_--_TEST_ONLY"
        assert not_specified_id_template.model_template_path == model_template_path
        # Check for template file with specified model_template_id
        id_specified_model_path = TestHyperParameterData.get_path_to_file(r"./id_specified_template.yaml")
        model_id = "Parsed_Model_ID_1"
        with open(model_template_path) as model_template_file:
            id_specified_template_content = yaml.safe_load(model_template_file)
            id_specified_template_content["model_template_id"] = model_id
        with open(id_specified_model_path, "w") as new_template:
            new_template.write(yaml.dump(id_specified_template_content))
        id_specified_template = parse_model_template(id_specified_model_path)
        assert id_specified_template.model_template_id == model_id
        assert id_specified_template.model_template_path == id_specified_model_path
        remove(id_specified_model_path)
        # Check ValueError exception raised if model template is list-type
        incorrect_model_template_path = TestHyperParameterData.get_path_to_file(r"./incorrect_model_template.yaml")
        with open(incorrect_model_template_path, "w+") as incorrect_yaml_file:
            incorrect_yaml_file.write("[]")
        with pytest.raises(ValueError):
            parse_model_template(incorrect_model_template_path)
        remove(incorrect_model_template_path)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_parse_model_template_from_dict(self):
        """
        <b>Description:</b>
        Check parse_model_template_from_dict function returns expected instance of ModelTemplate class
        <b>Expected results:</b>
        Test passes if parse_model_template_from_dict function returns instance of ModelTemplate class
        <b>Steps</b>
        1. Check ModelTemplate instance returned by test_parse_model_template_from_dict function for dictionary
        with specified model_template_id and model_template_path
        parameters
        """
        model_template = CommonMethods.cut_parameter_overrides_from_model_template()
        model_template_path = model_template.get("new_config_path")
        override_parameters = model_template.get("override_parameters")
        hyper_parameters = HyperParameterData(
            base_path=TestHyperParameterData().config_path(),
            parameter_overrides=override_parameters,
        )
        hyper_parameters.load_parameters(model_template_path)
        template_dictionary = {
            "model_template_id": "Dictionary_Model_1",
            "model_template_path": model_template_path,
            "name": "Custom Object Detection -- TEST ONLY",
            "task_type": TaskType.DETECTION,
            "task_family": TaskFamily.VISION,
            "instantiation": InstantiationType.CLASS,
            "summary": "Fast and lightweight object detector.",
            "application": None,
            "framework": "Test framework",
            "entrypoints": EntryPoints(base="base interface", openvino="OpenVINO interface"),
            "hyper_parameters": hyper_parameters,
            "max_nodes": 1,
            "training_targets": [TargetDevice.GPU, TargetDevice.CPU],
            "inference_targets": [
                TargetDevice.CPU,
                TargetDevice.GPU,
                TargetDevice.VPU,
            ],
        }
        model_template_from_dictionary = parse_model_template_from_dict(template_dictionary)
        CommonMethods.check_model_attributes(model_template_from_dictionary, template_dictionary)
        remove(model_template_path)
