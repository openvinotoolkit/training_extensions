# Copyright (C) 2021 Intel Corporation
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

from pathlib import Path

import pytest
import yaml

from otx.api.configuration import ConfigurableParameters, cfg_helper
from otx.api.entities.id import ID
from otx.api.entities.label import Domain, LabelEntity
from otx.api.entities.label_schema import LabelSchemaEntity
from otx.api.entities.model_template import parse_model_template
from otx.api.entities.task_environment import TaskEnvironment
from tests.test_helpers import ConfigExample
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


def __get_path_to_file(filename: str):
    """
    Return the path to the file named 'filename', which lives in the tests/entities directory
    """
    return str(Path(__file__).parent / Path(filename))


def dummy_config():
    """
    Return dict from yaml file
    """
    cur_config = __get_path_to_file("./dummy_config.yaml")
    with open(cur_config, "r") as stream:
        return yaml.safe_load(stream)


def environment():
    """
    Return TaskEnvironment
    """
    car = LabelEntity(id=ID(123456789), name="car", domain=Domain.DETECTION, is_empty=False)
    person = LabelEntity(id=ID(987654321), name="person", domain=Domain.DETECTION, is_empty=False)
    labels_list = [car, person]
    dummy_template = __get_path_to_file("./dummy_template.yaml")
    model_template = parse_model_template(dummy_template)
    hyper_parameters = model_template.hyper_parameters.data
    params = cfg_helper.create(hyper_parameters)
    labels_schema = LabelSchemaEntity.from_labels(labels_list)
    environment = TaskEnvironment(
        model=None,
        hyper_parameters=params,
        label_schema=labels_schema,
        model_template=model_template,
    )
    return environment


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTaskEnvironment:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_task_environment(self):
        """
        <b>Description:</b>
        Check the TaskEnvironment can correctly return the value

        <b>Input data:</b>
        Dummy data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Using an already created dummy environment.
        2. Checking class fields
        """

        env = environment()
        __dummy_config = dummy_config()

        assert env == TaskEnvironment(
            model=None,
            model_template=env.model_template,
            hyper_parameters=env.get_hyper_parameters(),
            label_schema=env.label_schema,
        )
        assert isinstance(env, TaskEnvironment)
        assert env != "Fail params"
        assert len(env.get_labels()) == 2

        for i in ["header", "description", "visible_in_ui"]:
            assert getattr(env.get_model_configuration().configurable_parameters, i) == __dummy_config[i]

        assert env.get_model_configuration().configurable_parameters.id == ID()

        for param in __dummy_config:
            getattr(env.get_hyper_parameters(), param) == __dummy_config[param]

        assert env.get_hyper_parameters().id == ID()

        assert "model=None" in repr(env)
        assert "label_schema=LabelSchemaEntity(label_groups=[LabelGroup(id=" in repr(env)
        assert "name=from_label_list" in repr(env)
        assert "group_type=LabelGroupType.EXCLUSIVE" in repr(env)
        assert "labels=[LabelEntity(123456789, name=car, hotkey=, domain=DETECTION" in repr(env)
        assert "LabelEntity(987654321, name=person, hotkey=, domain=DETECTION" in repr(env)
        assert "CONFIGURABLE_PARAMETERS(header='Configuration for an object detection task -- TEST ONLY'" in repr(env)
        assert "description='Configuration for an object detection task -- TEST ONLY'" in repr(env)
        assert "visible_in_ui=True" in repr(env)
        assert "id=ID()" in repr(env)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_compare_learning_parameters_num_workers(self):
        """
        <b>Description:</b>
        Check matches ConfigExample and dummy_config.yaml num_workers default_value

        <b>Input data:</b>
        ConfigExample, dummy_config.yaml

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking num_workers default_value from ConfigExample and dummy_config.yaml
        """

        env = environment()
        config_example = env.get_hyper_parameters(ConfigExample)

        env_learning_parameters_num_workers = (
            env.get_hyper_parameters().learning_parameters.num_workers
        )  # "default_value"
        config_example_learning_parameters_num_workers = (
            config_example.learning_parameters._default.factory.num_workers.metadata["default_value"]
        )

        # From dummy_config.yaml because it is missing in dummy_template.yaml "parameter_overrides"
        assert env_learning_parameters_num_workers == 0
        assert config_example_learning_parameters_num_workers == 4  # From ConfigExample

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_compare_learning_parameters_batch_size(self):
        """
        <b>Description:</b>
        Check matches ConfigExample and dummy_config.yaml batch_size default_value

        <b>Input data:</b>
        ConfigExample, dummy_config.yaml

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking batch_size from ConfigExample and dummy_config.yaml
        """

        env = environment()
        config_example = env.get_hyper_parameters(ConfigExample)

        env_learning_parameters_batch_size = (
            env.get_hyper_parameters().learning_parameters.batch_size
        )  # "default_value"
        config_example_learning_parameters_batch_size = (
            config_example.learning_parameters._default.factory.batch_size.metadata["default_value"]
        )

        assert env_learning_parameters_batch_size == 64  # From dummy_template.yaml "parameter_overrides"
        assert config_example_learning_parameters_batch_size == 5  # From ConfigExample

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_compare_learning_parameters_num_iters(self):
        """
        <b>Description:</b>
        Check matches ConfigExample and dummy_config.yaml num_iters default_value

        <b>Input data:</b>
        ConfigExample, dummy_config.yaml

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking num_iters from ConfigExample and dummy_config.yaml
        """

        env = environment()
        config_example = env.get_hyper_parameters(ConfigExample)

        env_learning_parameters_num_iters = env.get_hyper_parameters().learning_parameters.num_iters  # "default_value"
        config_example_learning_parameters_num_iters = (
            config_example.learning_parameters._default.factory.num_iters.metadata["default_value"]
        )

        assert env_learning_parameters_num_iters == 13000  # From dummy_template.yaml "parameter_overrides"
        assert config_example_learning_parameters_num_iters == 1  # From ConfigExample

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_compare_learning_parameters_learning_rate(self):
        """
        <b>Description:</b>
        Check matches ConfigExample and dummy_config.yaml learning_rate default_value

        <b>Input data:</b>
        ConfigExample, dummy_config.yaml

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking learning_rate from ConfigExample and dummy_config.yaml
        """

        env = environment()
        config_example = env.get_hyper_parameters(ConfigExample)

        env_learning_parameters_learning_rate = (
            env.get_hyper_parameters().learning_parameters.learning_rate
        )  # "default_value"
        config_example_learning_parameters_learning_rate = (
            config_example.learning_parameters._default.factory.learning_rate.metadata["default_value"]
        )

        assert env_learning_parameters_learning_rate == 0.05  # From dummy_template.yaml "parameter_overrides"
        assert config_example_learning_parameters_learning_rate == 0.01  # From ConfigExample

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_compare_learning_parameters_num_checkpoints(self):
        """
        <b>Description:</b>
        Check matches ConfigExample and dummy_config.yaml num_checkpoints default_value
        "num_checkpoints" is missing in ConfigExample

        <b>Input data:</b>
        ConfigExample, dummy_config.yaml

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking num_checkpoints from ConfigExample and dummy_config.yaml
        """

        env = environment()
        config_example = env.get_hyper_parameters(ConfigExample)

        env_learning_parameters_num_checkpoints = (
            env.get_hyper_parameters().learning_parameters.num_checkpoints
        )  # "default_value"

        # From dummy_config.yaml because it is missing in dummy_template.yaml "parameter_overrides"
        assert env_learning_parameters_num_checkpoints == 5

        # Attempt to access the missing parameter in ConfigExample
        with pytest.raises(AttributeError):
            # AttributeError: type object '__LearningParameters' has no attribute 'num_checkpoints'
            config_example.learning_parameters._default.factory.num_checkpoints.metadata["default_value"]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_dummy_config_missing_param(self):
        """
        <b>Description:</b>
        Check dummy_config missing_param

        <b>Input data:</b>
        dummy_config

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking missing_param dummy_config
        """

        env = environment()

        # Attempt to access the missing parameter in dummy_config.yaml
        with pytest.raises(AttributeError):
            # AttributeError: 'PARAMETER_GROUP' object has no attribute 'missing_param'
            env.get_hyper_parameters().learning_parameters.missing_param

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_compare_postprocessing_confidence_threshold(self):
        """
        <b>Description:</b>
        Check matches ConfigExample and dummy_config.yaml confidence_threshold default_value

        <b>Input data:</b>
        ConfigExample, dummy_config.yaml

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking confidence_threshold from ConfigExample and dummy_config.yaml
        """

        env = environment()
        config_example = env.get_hyper_parameters(ConfigExample)

        env_postprocessing_confidence_threshold = env.get_hyper_parameters().postprocessing.confidence_threshold
        cep = config_example.postprocessing
        config_example_postprocessing_confidence_threshold = cep._default.factory.confidence_threshold.metadata[
            "default_value"
        ]

        # From dummy_config.yaml because it is missing in dummy_template.yaml "parameter_overrides"
        assert env_postprocessing_confidence_threshold == 0.35
        assert config_example_postprocessing_confidence_threshold == 0.25  # From ConfigExample

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_compare_postprocessing_result_based_confidence_threshold(self):
        """
        <b>Description:</b>
        Check matches ConfigExample and dummy_config.yaml result_based_confidence_threshold default_value

        <b>Input data:</b>
        ConfigExample, dummy_config.yaml

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking result_based_confidence_threshold from ConfigExample and dummy_config.yaml
        """

        env = environment()
        config_example = env.get_hyper_parameters(ConfigExample)

        env_postprocessing_result_based_confidence_threshold = (
            env.get_hyper_parameters().postprocessing.result_based_confidence_threshold
        )  # "default_value"
        cep = config_example.postprocessing
        def_factory = cep._default.factory
        rbct = def_factory.result_based_confidence_threshold
        config_example_postprocessing_result_based_confidence_threshold = rbct.metadata["default_value"]

        # From dummy_config.yaml because it is missing in dummy_template.yaml "parameter_overrides"
        assert env_postprocessing_result_based_confidence_threshold is True
        assert config_example_postprocessing_result_based_confidence_threshold is True  # From ConfigExample

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_set_hyper_parameters(self):
        """
        <b>Description:</b>
        Check set_hyper_parameters() method

        <b>Input data:</b>
        Dummmy data

        <b>Expected results:</b>
        Test passes if incoming data is processed correctly

        <b>Steps</b>
        1. Checking parameters after setting
        """
        env = environment()

        header = "Test header"
        description = "Test description"
        visible_in_ui = False
        id = ID(123456789)

        hyper_parameters = ConfigurableParameters(
            header=header, description=description, visible_in_ui=visible_in_ui, id=id
        )
        env.set_hyper_parameters(hyper_parameters=hyper_parameters)
        assert env.get_hyper_parameters().header == header
        assert env.get_hyper_parameters().description == description
        assert env.get_hyper_parameters().visible_in_ui == visible_in_ui
        assert env.get_hyper_parameters().id == id

        assert env.get_model_configuration().configurable_parameters.header == header
        assert env.get_model_configuration().configurable_parameters.description == description
        assert env.get_model_configuration().configurable_parameters.visible_in_ui == visible_in_ui
        assert env.get_model_configuration().configurable_parameters.id == id

        with pytest.raises(ValueError):
            # ValueError: Unable to set hyper parameters, invalid input: 123
            env.set_hyper_parameters(hyper_parameters="123")
