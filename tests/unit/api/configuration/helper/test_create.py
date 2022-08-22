# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from attr import _make
from omegaconf import DictConfig

from otx.api.configuration.configurable_parameters import ConfigurableParameters
from otx.api.configuration.elements import ConfigurableEnum
from otx.api.configuration.elements.parameter_group import ParameterGroup
from otx.api.configuration.enums.config_element_type import ConfigElementType
from otx.api.configuration.enums.model_lifecycle import ModelLifecycle
from otx.api.configuration.helper.config_element_mapping import GroupElementMapping
from otx.api.configuration.helper.create import (
    construct_attrib_from_dict,
    construct_ui_rules_from_dict,
    contains_parameter_groups,
    create,
    create_default_configurable_enum_from_dict,
    create_nested_parameter_group,
    from_dict_attr,
    gather_parameter_arguments_and_values_from_dict,
)
from otx.api.configuration.ui_rules.rules import NullUIRules, Rule, UIRules
from otx.api.configuration.ui_rules.types import Action, Operator
from tests.unit.api.configuration.dummy_config import SomeEnumSelectable
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestCreateFunctions:
    @staticmethod
    def int_rule_dict():
        return {
            "parameter": "int parameter",
            "value": 100,
            "operator": Operator.LESS_THAN,
            "type": "RULE",
        }

    def int_ui_rules_params(self):
        return {"rules": [self.int_rule_dict()], "type": "UI_RULES"}

    def non_selectable_dict(self):
        return {
            "default_value": 70,
            "header": "non selectable parameter header",
            "ui_rules": self.int_ui_rules_params(),
            "type": "INTEGER",
        }

    @staticmethod
    def selectable_dict():
        return DictConfig(
            content={
                "header": "selectable parameter header",
                "enum_name": "test enum",
                "options": ["test_1", "test_2"],
                "default_value": 2,
                "type": "SELECTABLE",
                "value": 1,
                "affects_outcome_of": ModelLifecycle.TESTING,
            }
        )

    @staticmethod
    def check_ui_rules(
        ui_rules,
        expected_rules,
        expected_action=Action.DISABLE_EDITING,
        expected_operator=Operator.AND,
    ):
        assert isinstance(ui_rules, UIRules)
        assert ui_rules.rules == expected_rules
        assert ui_rules.action == expected_action
        assert ui_rules.operator == expected_operator
        assert ui_rules.type == ConfigElementType.UI_RULES

    def nested_config_dict_section(self):
        non_nested_parameters = {
            "header": "non-nested parameter header",
            "non_selectable": self.non_selectable_dict(),
            "type": GroupElementMapping.CONFIGURABLE_PARAMETERS,
        }
        selectable_parameters = {
            "header": "nested parameter header",
            "selectable": self.selectable_dict(),
            "type": GroupElementMapping.CONFIGURABLE_PARAMETERS,
        }

        nested_parameters = {
            "header": "nested parameters group header",
            "selectable": selectable_parameters,
            "type": GroupElementMapping.PARAMETER_GROUP,
        }

        return {
            "header": "test header",
            "non_nested": non_nested_parameters,
            "nested": nested_parameters,
            "type": GroupElementMapping.CONFIGURABLE_PARAMETERS,
        }

    @staticmethod
    def check_parameter_group(parameter_group, expected_type):
        # Checking parameter group attributes
        assert isinstance(parameter_group, expected_type)
        assert parameter_group.description == "Default parameter group description"
        assert parameter_group.groups == ["nested", "non_nested"]
        assert parameter_group.header == "test header"
        assert parameter_group.parameters == []
        assert parameter_group.type == ConfigElementType.CONFIGURABLE_PARAMETERS
        # Checking non-nested configurable parameter
        non_nested = parameter_group.non_nested
        assert isinstance(non_nested, ConfigurableParameters)
        assert non_nested.description == "Default parameter group description"
        assert non_nested.groups == []
        assert non_nested.header == "non-nested parameter header"
        assert non_nested.non_selectable == 70  # type: ignore
        assert non_nested.parameters == ["non_selectable"]
        assert non_nested.type == ConfigElementType.CONFIGURABLE_PARAMETERS
        # Checking nested parameter group
        nested = parameter_group.nested
        assert isinstance(nested, ParameterGroup)
        assert nested.groups == ["selectable"]
        assert nested.header == "nested parameters group header"
        assert nested.parameters == []
        assert nested.type == ConfigElementType.PARAMETER_GROUP
        # Checking nested parameter
        parameter = nested.selectable  # type: ignore
        assert isinstance(parameter, ConfigurableParameters)
        assert parameter.description == "Default parameter group description"
        assert parameter.groups == []
        assert parameter.header == "nested parameter header"
        assert parameter.selectable == 1  # type: ignore
        assert parameter.parameters == ["selectable"]
        assert parameter.type == ConfigElementType.CONFIGURABLE_PARAMETERS

    @staticmethod
    def check_non_nested_configurable_parameters(non_nested):
        assert isinstance(non_nested, ConfigurableParameters)
        assert non_nested.description == "Default parameter group description"
        assert non_nested.groups == []
        assert non_nested.header == "non-nested parameter header"
        assert non_nested.non_selectable == 70  # type: ignore
        assert non_nested.parameters == ["non_selectable"]
        assert non_nested.type == ConfigElementType.CONFIGURABLE_PARAMETERS

    @staticmethod
    def check_nested_parameter_group(nested):
        assert isinstance(nested, ParameterGroup)
        assert nested.groups == ["selectable"]
        assert nested.header == "nested parameters group header"
        assert nested.parameters == []
        assert nested.type == ConfigElementType.PARAMETER_GROUP
        # Checking nested parameter
        parameter = nested.selectable  # type: ignore
        assert isinstance(parameter, ConfigurableParameters)
        assert parameter.description == "Default parameter group description"
        assert parameter.groups == []
        assert parameter.header == "nested parameter header"
        assert parameter.selectable == 1  # type: ignore
        assert parameter.parameters == ["selectable"]
        assert parameter.type == ConfigElementType.CONFIGURABLE_PARAMETERS

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_construct_attrib_from_dict(self):
        """
        <b>Description:</b>
        Check "construct_attrib_from_dict" function

        <b>Input data:</b>
        "dict_object" dictionary or DictConfig parameter

        <b>Expected results:</b>
        Test passes if object returned by "construct_attrib_from_dict" function is equal to expected

        <b>Steps</b>
        1. Check Rule object returned by "construct_attrib_from_dict" function
        2. Check UIRules object returned by "construct_attrib_from_dict" function
        3. Check that ValueError exception is raised when unexpected "type" is specified in "dict_object" parameter
        """

        def check_rule(
            rule,
            expected_parameter,
            expected_value,
            expected_operator=Operator.EQUAL_TO,
        ):
            assert isinstance(rule, Rule)
            assert rule.parameter == expected_parameter
            assert rule.value == expected_value
            assert rule.type == ConfigElementType.RULE
            assert rule.operator == expected_operator

        # Checking Rule object returned by "construct_attrib_from_dict"
        # Constructing Rule with default optional parameters
        params = {
            "parameter": "string parameter",
            "value": "some string",
            "type": "RULE",
        }
        for dict_object in [params, DictConfig(content=params)]:
            check_rule(
                rule=construct_attrib_from_dict(dict_object=dict_object),
                expected_parameter="string parameter",
                expected_value="some string",
            )
        # Constructing Rule with specified optional parameters
        params = {
            "parameter": "int parameter",
            "value": 5,
            "operator": Operator.LESS_THAN,
            "type": "RULE",
        }
        for dict_object in [params, DictConfig(content=params)]:
            check_rule(
                rule=construct_attrib_from_dict(dict_object=dict_object),
                expected_parameter="int parameter",
                expected_value=5,
                expected_operator=Operator.LESS_THAN,
            )
        # Checking UIRules object returned by "construct_attrib_from_dict"
        rules = [
            {"parameter": "string parameter", "value": "some string"},
            {"parameter": "int parameter", "value": 2},
        ]
        # Constructing UIRules with default optional parameters
        params = {"rules": rules, "type": "UI_RULES"}
        for dict_object in [params, DictConfig(content=params)]:
            self.check_ui_rules(
                ui_rules=construct_attrib_from_dict(dict_object=dict_object),
                expected_rules=rules,
            )
        # Constructing UIRules with specified optional parameters
        params = {
            "rules": rules,
            "action": Action.ENABLE_EDITING,
            "operator": Operator.EQUAL_TO,
            "type": "UI_RULES",
        }
        for dict_object in [params, DictConfig(content=params)]:
            self.check_ui_rules(
                ui_rules=construct_attrib_from_dict(dict_object=dict_object),
                expected_rules=rules,
                expected_action=Action.ENABLE_EDITING,
                expected_operator=Operator.EQUAL_TO,
            )
        # Checking that ValueError exception is raised when unexpected "type" is specified in "dict_object"
        params = {
            "parameter": "string parameter",
            "value": "some string",
            "type": "unexpected type",
        }
        with pytest.raises(ValueError):
            construct_attrib_from_dict(dict_object=params)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_construct_ui_rules_from_dict(self):
        """
        <b>Description:</b>
        Check "construct_ui_rules_from_dict" function

        <b>Input data:</b>
        "ui_exposure_settings" dictionary or DictConfig parameter

        <b>Expected results:</b>
        Test passes if UIRules object returned by "construct_ui_rules_from_dict" function is equal to expected

        <b>Steps</b>
        1. Check UIRules object returned by "construct_ui_rules_from_dict" function when several "RULE" objects are
        specified in "rules" key of "ui_exposure_settings" parameter
        2. Check UIRules object returned by "construct_ui_rules_from_dict" function when several "RULE" objects and
        nested "UI_RULE" are specified in "rules" key of "ui_exposure_settings" parameter
        3. Check that NullUIRules object is returned by "construct_ui_rules_from_dict" function when None is specified
        as "ui_exposure_settings" parameter
        4. Check that NullUIRules object is returned by "construct_ui_rules_from_dict" function when empty list is
        specified in "rules" key of "ui_exposure_settings" parameter
        5. Check that ValueError exception is raised when one of objects specified in "rules" key of
        "ui_exposure_settings" parameter has unexpected "TYPE"
        """
        # Checking UIRules returned by "construct_ui_rules_from_dict" when several "RULE" are specified in "rules" of
        # "ui_exposure_settings"
        rule = self.int_rule_dict()
        other_rule = {"parameter": "bool parameter", "value": False, "type": "RULE"}
        params = {"rules": [rule, other_rule], "type": "UI_RULES"}
        expected_rules = [
            Rule(parameter="int parameter", value=100, operator=Operator.LESS_THAN),
            Rule(parameter="bool parameter", value=False),
        ]
        for ui_exposure_settings in [params, DictConfig(content=dict(params))]:
            self.check_ui_rules(
                ui_rules=construct_ui_rules_from_dict(ui_exposure_settings=ui_exposure_settings),
                expected_rules=expected_rules,
            )
        # Checking UIRules returned by "construct_ui_rules_from_dict" when several "RULE" and "UI_RULE" are specified in
        # "rules" of "ui_exposure_settings"
        nested_rule = {
            "parameter": "nested float parameter",
            "value": 10.6,
            "operator": Operator.GREATER_THAN,
            "type": "RULE",
        }

        other_nested_rule = {
            "parameter": "nested bool parameter",
            "value": True,
            "type": "RULE",
        }
        nested_ui_rules = {
            "rules": [nested_rule, other_nested_rule],
            "action": Action.ENABLE_EDITING,
            "operator": Operator.EQUAL_TO,
            "type": "UI_RULES",
        }
        params = {"rules": [rule, other_rule, nested_ui_rules], "type": "UI_RULES"}
        expected_nested_rules = [
            Rule(
                parameter="nested float parameter",
                value=10.6,
                operator=Operator.GREATER_THAN,
            ),
            Rule(parameter="nested bool parameter", value=True),
        ]
        expected_rules = [
            Rule(parameter="int parameter", value=100, operator=Operator.LESS_THAN),
            Rule(parameter="bool parameter", value=False),
            UIRules(
                rules=expected_nested_rules,
                action=Action.ENABLE_EDITING,
                operator=Operator.EQUAL_TO,
            ),
        ]
        for ui_exposure_settings in [params, DictConfig(content=dict(params))]:
            self.check_ui_rules(
                ui_rules=construct_ui_rules_from_dict(ui_exposure_settings=ui_exposure_settings),
                expected_rules=expected_rules,
            )
        # Checking that NullUIRules returned by "construct_ui_rules_from_dict" when None is specified as
        # "ui_exposure_settings"
        assert construct_ui_rules_from_dict(ui_exposure_settings=None) == NullUIRules()  # type: ignore
        # Checking that NullUIRules returned by "construct_ui_rules_from_dict" when empty list is specified in "rules"
        # key of "ui_exposure_settings"
        params = {"rules": [], "type": "UI_RULES"}
        assert construct_ui_rules_from_dict(ui_exposure_settings=params) == NullUIRules()
        # Checking that ValueError exception is raised when one of objects specified in "rules" key of
        # "ui_exposure_settings" has unexpected "TYPE"
        invalid_rule = {
            "parameter": "invalid rule",
            "value": False,
            "type": "unexpected type",
        }
        params = {"rules": [rule, invalid_rule], "type": "UI_RULES"}
        with pytest.raises(ValueError):
            construct_ui_rules_from_dict(params)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_create_default_configurable_enum_from_dict(self):
        """
        <b>Description:</b>
        Check "create_default_configurable_enum_from_dict" function

        <b>Input data:</b>
        "parameter_dict" dictionary or DictConfig parameter

        <b>Expected results:</b>
        Test passes if dictionary returned by "create_default_configurable_enum_from_dict" function is equal to expected

        <b>Steps</b>
        1. Check dictionary returned by "create_default_configurable_enum_from_dict" function when ConfigurableEnum
        element is specified as "default_value" key value of "parameter_dict" parameter
        2. Check dictionary returned by "create_default_configurable_enum_from_dict" function when int object is
        specified as "default_value" key value of "parameter_dict" parameter
        3. Check that TypeError exception is raised when DictConfig object with "content" attribute equal to None is
        specified as "default_value" key value of "parameter_dict" parameter
        """
        # Checking dictionary returned by "create_default_configurable_enum_from_dict" when ConfigurableEnum element is
        # specified as "default_value" key value of "parameter_dict"
        params = {
            "enum_name": "test enum",
            "options": ["test_1", "test_2"],
            "default_value": SomeEnumSelectable.TEST_NAME1,
        }
        for parameter_dict in [params, DictConfig(content=dict(params))]:
            configurable_enum = create_default_configurable_enum_from_dict(parameter_dict=parameter_dict)
            assert isinstance(configurable_enum, dict)
            assert len(configurable_enum) == 1
            default_value = configurable_enum.get("default_value")
            assert default_value.name == "TEST_NAME1"
            assert default_value.value == "test_name_1"
            assert type(default_value) == SomeEnumSelectable
        # Checking dictionary returned by "create_default_configurable_enum_from_dict" when int object is specified as
        # "default_value" key value of "parameter_dict"
        params = {
            "enum_name": "test enum",
            "options": ["test_1", "test_2"],
            "default_value": 2,
        }
        for parameter_dict in [params, DictConfig(content=dict(params))]:
            configurable_enum = create_default_configurable_enum_from_dict(parameter_dict=parameter_dict)
            assert isinstance(configurable_enum, dict)
            assert len(configurable_enum) == 1
            default_value = configurable_enum.get("default_value")
            assert default_value.name == "test_2"
            assert default_value.value == 2
            default_value_type = type(default_value)
            assert default_value_type.__name__ == "test enum"
            assert issubclass(default_value_type, ConfigurableEnum)
            assert len(default_value_type) == 2
            assert default_value_type.test_1.name == "test_1"
            assert default_value_type.test_1.value == 1
            assert default_value_type.test_2.name == "test_2"
            assert default_value_type.test_2.value == 2
        # Checking that TypeError exception is raised when DictConfig with "content" equal to None is specified as
        # "default_value" key value of "parameter_dict"
        with pytest.raises(TypeError):
            create_default_configurable_enum_from_dict(parameter_dict=DictConfig(content=None))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_gather_parameter_arguments_and_values_from_dict(self):
        """
        <b>Description:</b>
        Check "gather_parameter_arguments_and_values_from_dict" function

        <b>Input data:</b>
        "config_dict_section" dictionary or DictConfig parameter

        <b>Expected results:</b>
        Test passes if dictionary returned by "gather_parameter_arguments_and_values_from_dict" function is equal to
        expected

        <b>Steps</b>
        1. Check dictionary returned by "gather_parameter_arguments_and_values_from_dict" function when INTEGER and
        SELECTABLE parameter dictionaries and non-dictionary value are specified in "config_dict_section" parameter
        2. Check that ValueError exception is raised when one of parameters contains "type" key equal to None
        """
        # Checking dictionary returned by "gather_parameter_arguments_and_values_from_dict" when INTEGER and
        # SELECTABLE argument dictionaries and non-dictionary value are specified in "config_dict_section"
        non_selectable_dict = self.non_selectable_dict()
        selectable_dict = self.selectable_dict()
        params = {
            "non_selectable": non_selectable_dict,
            "selectable": selectable_dict,
            "non_dict_key": 6,
            "type": "non required type",
        }
        for config_dict_section in [params, DictConfig(content=dict(params))]:
            arguments_and_values_dict = gather_parameter_arguments_and_values_from_dict(
                config_dict_section=config_dict_section
            )
            assert len(arguments_and_values_dict) == 3
            # Checking "make_arguments" key
            assert len(arguments_and_values_dict.get("make_arguments")) == 2
            # Checking non-Selectable argument
            argument = arguments_and_values_dict.get("make_arguments").get("non_selectable")
            assert isinstance(argument, _make._CountingAttr)
            assert argument.type == int
            metadata = argument.metadata
            assert metadata.get("default_value") == 70
            assert metadata.get("description") == "Default integer description"
            assert metadata.get("header") == "non selectable parameter header"
            assert metadata.get("affects_outcome_of") == ModelLifecycle.NONE
            assert metadata.get("type") == ConfigElementType.INTEGER
            assert metadata.get("ui_rules") == construct_ui_rules_from_dict(dict(self.int_ui_rules_params()))
            # Checking Selectable argument
            argument = arguments_and_values_dict.get("make_arguments").get("selectable")
            assert isinstance(argument, _make._CountingAttr)
            assert argument.type == ConfigurableEnum
            metadata = argument.metadata
            assert type(metadata.get("default_value")).__name__ == "test enum"
            assert metadata.get("default_value").name == "test_2"
            assert metadata.get("default_value").value == 2
            assert metadata.get("description") == "Default selectable description"
            assert metadata.get("header") == "selectable parameter header"
            assert metadata.get("affects_outcome_of") == ModelLifecycle.TESTING
            assert metadata.get("type") == ConfigElementType.SELECTABLE
            assert metadata.get("ui_rules") == NullUIRules()
            # Checking "call_arguments" key
            assert arguments_and_values_dict.get("call_arguments") == {"non_dict_key": 6}
            # Checking "values" key
            assert len(arguments_and_values_dict.get("values")) == 2
            assert arguments_and_values_dict.get("values").get("selectable") == 1
            assert not arguments_and_values_dict.get("values")["non_selectable"]
        # Checking that ValueError exception is raised when one of parameters contains "type" key equal to None
        non_selectable_dict.pop("type")
        config_dict_section = {
            "no_type_parameter": non_selectable_dict,
            "selectable": selectable_dict,
        }
        with pytest.raises(ValueError):
            gather_parameter_arguments_and_values_from_dict(config_dict_section=config_dict_section)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_contains_parameter_groups(self):
        """
        <b>Description:</b>
        Check "contains_parameter_groups" function

        <b>Input data:</b>
        "config_dict" dictionary or DictConfig parameter

        <b>Expected results:</b>
        Test passes if list returned by "contains_parameter_groups" function is equal to expected
        """
        configurable_parameters = {
            "header": "int group header",
            "int_value": 10,
            "type": GroupElementMapping.CONFIGURABLE_PARAMETERS,
        }
        parameter_group = {
            "header": "bool group header",
            "bool_value": False,
            "type": GroupElementMapping.PARAMETER_GROUP,
        }

        unexpected_type_group = {
            "header": "unexpected type header",
            "value": "some value",
            "type": "unexpected type",
        }

        parameters = {
            "configurable_parameters_test_group": configurable_parameters,
            "parameter_test_group": parameter_group,
            "unexpected_type_test_group": unexpected_type_group,
        }

        for config_dict in [parameters, DictConfig(content=dict(parameters))]:
            assert contains_parameter_groups(config_dict=config_dict) == [
                "configurable_parameters_test_group",
                "parameter_test_group",
            ]

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_create_nested_parameter_group(self):
        """
        <b>Description:</b>
        Check "create_nested_parameter_group" function

        <b>Input data:</b>
        "config_dict_section" dictionary or DictConfig parameter

        <b>Expected results:</b>
        Test passes if ParameterGroup object returned by "create_nested_parameter_group" function is equal to expected
        """
        parameters = self.nested_config_dict_section()
        for config_dict_section in parameters, DictConfig(content=dict(parameters)):
            nested_parameter_group = create_nested_parameter_group(config_dict_section=config_dict_section)
            self.check_parameter_group(
                parameter_group=nested_parameter_group,  # type: ignore
                expected_type=ParameterGroup,
            )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_from_dict_attr(self):
        """
        <b>Description:</b>
        Check "from_dict_attr" function

        <b>Input data:</b>
        "config_dict" dictionary or DictConfig parameter

        <b>Expected results:</b>
        Test passes if ConfigurableParameters returned by "from_dict_attr" function is equal to expected
        """
        parameters = self.nested_config_dict_section()
        for config_dict in [parameters, DictConfig(content=dict(parameters))]:
            parameter_group = from_dict_attr(config_dict=config_dict)
            self.check_parameter_group(
                parameter_group=parameter_group,  # type: ignore
                expected_type=ConfigurableParameters,
            )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_create(self):
        """
        <b>Description:</b>
        Check "create" function

        <b>Input data:</b>
        "input_config" dictionary or DictConfig parameter

        <b>Expected results:</b>
        Test passes if ConfigurableParameters object returned by "from_dict_attr" function is equal to expected

        <b>Steps</b>
        1. Check ConfigurableParameters object returned by "from_dict_attr" function when dictionary is specified as
        "input_config" parameter
        2. Check ConfigurableParameters object returned by "from_dict_attr" function when string is specified as
        "input_config" parameter
        """
        # Checking ConfigurableParameters returned by "from_dict_attr" when dictionary is specified as "input_config"
        parameters = self.nested_config_dict_section()
        for input_config in [parameters, DictConfig(content=dict(parameters))]:
            parameter_group = create(input_config=input_config)
            self.check_parameter_group(
                parameter_group=parameter_group,  # type: ignore
                expected_type=ConfigurableParameters,
            )

        # Checking ConfigurableParameters returned by "from_dict_attr" when string is specified as "input_config"
        str_parameter = {
            "header": "str parameter header",
            "enum_name": "test enum",
            "options": ["test_1", "test_2"],
            "default_value": 2,
            "type": "SELECTABLE",
            "value": 1,
            "affects_outcome_of": "TESTING",
        }
        str_parameter_group = {
            "header": "str parameter group header",
            "str_parameter": str_parameter,
            "type": "PARAMETER_GROUP",
        }
        parameter_group = create(str(str_parameter_group))
        assert parameter_group.description == "Default parameter group description"
        assert parameter_group.groups == []
        assert parameter_group.header == "str parameter group header"
        assert parameter_group.parameters == ["str_parameter"]
        assert parameter_group.str_parameter == 1  # type: ignore
        assert parameter_group.type == ConfigElementType.PARAMETER_GROUP
