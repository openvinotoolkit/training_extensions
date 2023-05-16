# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from typing import Optional

import pytest
from attr import _make, validators

from otx.api.configuration.elements.configurable_enum import ConfigurableEnum
from otx.api.configuration.elements.primitive_parameters import (
    boolean_attribute,
    configurable_boolean,
    configurable_float,
    configurable_integer,
    float_selectable,
    selectable,
    set_common_metadata,
    string_attribute,
)
from otx.api.configuration.enums import AutoHPOState, ConfigElementType, ModelLifecycle
from otx.api.configuration.ui_rules import NullUIRules, Rule, UIRules
from tests.unit.api.configuration.dummy_config import SomeEnumSelectable
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestPrimitiveParameters:
    ui_rules = UIRules(rules=[Rule(parameter="rule parameter", value=1)])

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_set_common_metadata(self):
        """
        <b>Description:</b>
        Check "set_common_metadata" function

        <b>Input data:</b>
        "default_value" int, float, str, bool or ConfigurableEnum element, "header" string, "description" string,
        "warning" string, "editable" bool value, "affects_outcome_of" ModelLifecycle element, "ui_rules" UIRules object,
        "visible_in_ui" bool value, "parameter_type" ConfigElementType element

        <b>Expected results:</b>
        Test passes if dictionary returned by "set_common_metadata" function is equal to expected
        """
        header = "test header"
        description = "test description"
        warning = "test warning"
        editable = False
        affects_outcome_of = ModelLifecycle.TESTING
        ui_rules = self.ui_rules
        visible_in_ui = True
        parameter_type = ConfigElementType.CONFIGURABLE_PARAMETERS
        auto_hpo_state = AutoHPOState.POSSIBLE

        for default_value in [
            5,  # int "default_value"
            1.3,  # float "default_value"
            "default value string",  # str "default_value"
            False,  # bool "default_value"
            SomeEnumSelectable.TEST_2,  # ConfigurableEnum "default_value"
        ]:
            assert set_common_metadata(
                default_value=default_value,
                header=header,
                description=description,
                warning=warning,
                editable=editable,
                affects_outcome_of=affects_outcome_of,
                ui_rules=ui_rules,
                visible_in_ui=visible_in_ui,
                parameter_type=parameter_type,
                auto_hpo_state=auto_hpo_state,
                auto_hpo_value=default_value,
            ) == {
                "default_value": default_value,
                "description": description,
                "header": header,
                "warning": warning,
                "editable": editable,
                "visible_in_ui": visible_in_ui,
                "affects_outcome_of": affects_outcome_of,
                "ui_rules": ui_rules,
                "type": parameter_type,
                "auto_hpo_state": auto_hpo_state,
                "auto_hpo_value": default_value,
            }

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_configurable_integer(self):
        """
        <b>Description:</b>
        Check "configurable_integer" function

        <b>Input data:</b>
        "default_value" int, "header" string, "min_value" int, "max_value": int, "description" string, "warning" string,
        "editable" bool, "visible_in_ui" bool, "affects_outcome_of" ModelLifecycle element, "ui_rules" UIRules object

        <b>Expected results:</b>
        Test passes if _CountingAttr object returned by "configurable_integer" function is equal to expected

        <b>Steps</b>
        1. Check _CountingAttr object returned by "configurable_integer" function for default values of optional
        parameters
        2. Check _CountingAttr object returned by "configurable_integer" function for specified values of optional
        parameters
        """

        def check_configurable_integer(
            integer_instance,
            expected_min_value: int = 0,
            expected_max_value: int = 255,
            expected_description: str = "Default integer description",
            expected_warning: str = None,
            expected_editable: bool = True,
            expected_visible_in_ui: bool = True,
            expected_affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
            expected_ui_rules: UIRules = NullUIRules(),
            expected_auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
            expected_auto_hpo_value: int = None,
        ):
            expected_metadata = {
                "default_value": 100,
                "description": expected_description,
                "header": "configurable_integer header",
                "warning": expected_warning,
                "editable": expected_editable,
                "visible_in_ui": expected_visible_in_ui,
                "affects_outcome_of": expected_affects_outcome_of,
                "ui_rules": expected_ui_rules,
                "type": ConfigElementType.INTEGER,
                "min_value": expected_min_value,
                "max_value": expected_max_value,
                "auto_hpo_state": expected_auto_hpo_state,
                "auto_hpo_value": expected_auto_hpo_value,
            }
            assert isinstance(integer_instance, _make._CountingAttr)
            assert integer_instance._default == 100
            assert integer_instance.type == int
            assert len(integer_instance._validator._validators) == 2
            assert integer_instance.metadata == expected_metadata

        # Checking _CountingAttr object returned by "configurable_integer" for default values of optional parameters
        default_value = 100
        header = "configurable_integer header"
        actual_integer = configurable_integer(default_value=default_value, header=header)
        check_configurable_integer(integer_instance=actual_integer)  # type: ignore
        # Checking _CountingAttr object returned by "configurable_integer" for specified values of optional parameters
        min_value = 10
        max_value = 200
        description = "configurable_integer description"
        warning = "configurable_integer warning"
        editable = False
        visible_in_ui = False
        affects_outcome_of = ModelLifecycle.TESTING
        ui_rules = self.ui_rules
        auto_hpo_state = AutoHPOState.POSSIBLE
        auto_hpo_value = min_value

        actual_integer = configurable_integer(
            default_value=default_value,
            header=header,
            min_value=min_value,
            max_value=max_value,
            description=description,
            warning=warning,
            editable=editable,
            visible_in_ui=visible_in_ui,
            affects_outcome_of=affects_outcome_of,
            ui_rules=ui_rules,
            auto_hpo_value=auto_hpo_value,
            auto_hpo_state=auto_hpo_state,
        )
        check_configurable_integer(
            integer_instance=actual_integer,  # type: ignore
            expected_min_value=min_value,
            expected_max_value=max_value,
            expected_description=description,
            expected_warning=warning,
            expected_editable=editable,
            expected_visible_in_ui=visible_in_ui,
            expected_affects_outcome_of=affects_outcome_of,
            expected_ui_rules=ui_rules,
            expected_auto_hpo_state=auto_hpo_state,
            expected_auto_hpo_value=auto_hpo_value,
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_configurable_float(self):
        """
        <b>Description:</b>
        Check "configurable_float" function

        <b>Input data:</b>
        "default_value" float, "header" string, "min_value" float, "max_value": float, "description" string,
        "warning" string, "editable" bool, "visible_in_ui" bool, "affects_outcome_of" ModelLifecycle element,
        "ui_rules" UIRules object

        <b>Expected results:</b>
        Test passes if _CountingAttr object returned by "configurable_float" function is equal to expected

        <b>Steps</b>
        1. Check _CountingAttr object returned by "configurable_float" function for default values of optional
        parameters
        2. Check _CountingAttr object returned by "configurable_float" function for specified values of optional
        parameters
        """

        def check_configurable_float(
            float_instance,
            expected_min_value: float = 0.0,
            expected_max_value: float = 255.0,
            expected_step_size: Optional[float] = None,
            expected_description: str = "Default float description",
            expected_warning: str = None,
            expected_editable: bool = True,
            expected_visible_in_ui: bool = True,
            expected_affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
            expected_ui_rules: UIRules = NullUIRules(),
            expected_auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
            expected_auto_hpo_value: float = None,
        ):
            expected_metadata = {
                "default_value": 100.1,
                "description": expected_description,
                "header": "configurable_float header",
                "warning": expected_warning,
                "editable": expected_editable,
                "visible_in_ui": expected_visible_in_ui,
                "affects_outcome_of": expected_affects_outcome_of,
                "ui_rules": expected_ui_rules,
                "type": ConfigElementType.FLOAT,
                "min_value": expected_min_value,
                "max_value": expected_max_value,
                "step_size": expected_step_size,
                "auto_hpo_state": expected_auto_hpo_state,
                "auto_hpo_value": expected_auto_hpo_value,
            }
            assert isinstance(float_instance, _make._CountingAttr)
            assert float_instance._default == 100.1
            assert float_instance.type == float
            assert float_instance.metadata == expected_metadata

        # Checking _CountingAttr object returned by "configurable_float" for default values of optional parameters
        default_value = 100.1
        header = "configurable_float header"
        actual_float = configurable_float(default_value=default_value, header=header)
        check_configurable_float(float_instance=actual_float)  # type: ignore
        # Checking _CountingAttr object returned by "configurable_float" for specified values of optional parameters
        min_value = 0.1
        max_value = 160.2
        step_size = 0.3
        description = "configurable_float description"
        warning = "configurable_float warning"
        editable = False
        visible_in_ui = False
        affects_outcome_of = ModelLifecycle.TESTING
        ui_rules = self.ui_rules
        auto_hpo_state = AutoHPOState.POSSIBLE
        auto_hpo_value = min_value

        actual_float = configurable_float(
            default_value=default_value,
            header=header,
            min_value=min_value,
            max_value=max_value,
            step_size=step_size,
            description=description,
            warning=warning,
            editable=editable,
            visible_in_ui=visible_in_ui,
            affects_outcome_of=affects_outcome_of,
            ui_rules=ui_rules,
            auto_hpo_value=auto_hpo_value,
            auto_hpo_state=auto_hpo_state,
        )
        check_configurable_float(
            float_instance=actual_float,  # type: ignore
            expected_min_value=min_value,
            expected_max_value=max_value,
            expected_step_size=step_size,
            expected_description=description,
            expected_warning=warning,
            expected_editable=editable,
            expected_visible_in_ui=visible_in_ui,
            expected_affects_outcome_of=affects_outcome_of,
            expected_ui_rules=ui_rules,
            expected_auto_hpo_state=auto_hpo_state,
            expected_auto_hpo_value=auto_hpo_value,
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_configurable_boolean(self):
        """
        <b>Description:</b>
        Check "configurable_boolean" function

        <b>Input data:</b>
        "default_value" bool, "header" string, "description" string, "warning" string, "editable" bool,
        "visible_in_ui" bool, "affects_outcome_of" ModelLifecycle element, "ui_rules" UIRules object

        <b>Expected results:</b>
        Test passes if _CountingAttr object returned by "configurable_boolean" function is equal to expected

        <b>Steps</b>
        1. Check _CountingAttr object returned by "configurable_boolean" function for default values of optional
        parameters
        2. Check _CountingAttr object returned by "configurable_boolean" function for specified values of optional
        parameters
        """

        def check_configurable_boolean(
            boolean_instance,
            expected_description: str = "Default configurable boolean description",
            expected_warning: str = None,
            expected_editable: bool = True,
            expected_visible_in_ui: bool = True,
            expected_affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
            expected_ui_rules: UIRules = NullUIRules(),
            expected_auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
            expected_auto_hpo_value: bool = None,
        ):
            expected_metadata = {
                "default_value": True,
                "description": expected_description,
                "header": "configurable_boolean header",
                "warning": expected_warning,
                "editable": expected_editable,
                "visible_in_ui": expected_visible_in_ui,
                "affects_outcome_of": expected_affects_outcome_of,
                "ui_rules": expected_ui_rules,
                "type": ConfigElementType.BOOLEAN,
                "auto_hpo_state": expected_auto_hpo_state,
                "auto_hpo_value": expected_auto_hpo_value,
            }
            assert isinstance(boolean_instance, _make._CountingAttr)
            assert boolean_instance._default
            assert boolean_instance.type == bool
            assert isinstance(boolean_instance._validator, validators._InstanceOfValidator)  # type: ignore
            assert boolean_instance._validator.type == bool
            assert boolean_instance.metadata == expected_metadata

        # Checking _CountingAttr object returned by "configurable_boolean" for default values of optional parameters
        default_value = True
        header = "configurable_boolean header"
        actual_boolean = configurable_boolean(default_value=default_value, header=header)
        check_configurable_boolean(boolean_instance=actual_boolean)  # type: ignore
        # Checking _CountingAttr object returned by "configurable_boolean" for specified values of optional parameters
        description = "configurable_boolean description"
        warning = "configurable_boolean warning"
        editable = False
        visible_in_ui = False
        affects_outcome_of = ModelLifecycle.TESTING
        ui_rules = self.ui_rules
        auto_hpo_state = AutoHPOState.POSSIBLE
        auto_hpo_value = False

        actual_boolean = configurable_boolean(
            default_value=default_value,
            header=header,
            description=description,
            warning=warning,
            editable=editable,
            visible_in_ui=visible_in_ui,
            affects_outcome_of=affects_outcome_of,
            ui_rules=ui_rules,
            auto_hpo_value=auto_hpo_value,
            auto_hpo_state=auto_hpo_state,
        )
        check_configurable_boolean(
            boolean_instance=actual_boolean,  # type: ignore
            expected_description=description,
            expected_warning=warning,
            expected_editable=editable,
            expected_visible_in_ui=visible_in_ui,
            expected_affects_outcome_of=affects_outcome_of,
            expected_ui_rules=ui_rules,
            expected_auto_hpo_value=auto_hpo_value,
            expected_auto_hpo_state=auto_hpo_state,
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_float_selectable(self):
        """
        <b>Description:</b>
        Check "float_selectable" function

        <b>Input data:</b>
        "default_value" float, "header" string, "options" list, "description" string, "warning" string, "editable" bool,
        "visible_in_ui" bool, "affects_outcome_of" ModelLifecycle element, "ui_rules" UIRules object

        <b>Expected results:</b>
        Test passes if _CountingAttr object returned by "float_selectable" function is equal to expected

        <b>Steps</b>
        1. Check _CountingAttr object returned by "float_selectable" function for default values of optional parameters
        2. Check _CountingAttr object returned by "float_selectable" function for specified values of optional
        parameters
        """

        def check_float_selectable(
            float_selectable_instance,
            expected_description: str = "Default selectable description",
            expected_warning: str = None,
            expected_editable: bool = True,
            expected_visible_in_ui: bool = True,
            expected_affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
            expected_ui_rules: UIRules = NullUIRules(),
            expected_auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
            expected_auto_hpo_value: float = None,
        ):
            expected_metadata = {
                "default_value": 0.1,
                "description": expected_description,
                "header": "float_selectable header",
                "warning": expected_warning,
                "editable": expected_editable,
                "visible_in_ui": expected_visible_in_ui,
                "affects_outcome_of": expected_affects_outcome_of,
                "ui_rules": expected_ui_rules,
                "type": ConfigElementType.FLOAT_SELECTABLE,
                "options": [0.2, 1.4, 2.8],
                "auto_hpo_state": expected_auto_hpo_state,
                "auto_hpo_value": expected_auto_hpo_value,
            }
            assert isinstance(float_selectable_instance, _make._CountingAttr)
            assert float_selectable_instance._default
            assert float_selectable_instance.type == float
            assert float_selectable_instance.metadata == expected_metadata

        # Checking _CountingAttr object returned by "float_selectable" for default values of optional parameters
        default_value = 0.1
        header = "float_selectable header"
        options = [0.2, 1.4, 2.8]
        actual_float_selectable = float_selectable(default_value=default_value, options=options, header=header)
        check_float_selectable(float_selectable_instance=actual_float_selectable)  # type: ignore
        # Checking _CountingAttr object returned by "float_selectable" for specified values of optional parameters
        description = "float_selectable description"
        warning = "float_selectable warning"
        editable = False
        visible_in_ui = False
        affects_outcome_of = ModelLifecycle.TESTING
        ui_rules = self.ui_rules
        auto_hpo_state = AutoHPOState.POSSIBLE
        auto_hpo_value = options[-1]

        actual_float_selectable = float_selectable(
            default_value=default_value,
            options=options,
            header=header,
            description=description,
            warning=warning,
            editable=editable,
            visible_in_ui=visible_in_ui,
            affects_outcome_of=affects_outcome_of,
            ui_rules=ui_rules,
            auto_hpo_value=auto_hpo_value,
            auto_hpo_state=auto_hpo_state,
        )
        check_float_selectable(
            float_selectable_instance=actual_float_selectable,  # type: ignore
            expected_description=description,
            expected_warning=warning,
            expected_editable=editable,
            expected_visible_in_ui=visible_in_ui,
            expected_affects_outcome_of=affects_outcome_of,
            expected_ui_rules=ui_rules,
            expected_auto_hpo_value=auto_hpo_value,
            expected_auto_hpo_state=auto_hpo_state,
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_selectable(self):
        """
        <b>Description:</b>
        Check "selectable" function

        <b>Input data:</b>
        "default_value" ConfigurableEnum element, "header" string, "description" string, "warning" string, "editable"
        bool, "visible_in_ui" bool, "affects_outcome_of" ModelLifecycle element, "ui_rules" UIRules object

        <b>Expected results:</b>
        Test passes if _CountingAttr object returned by "selectable" function is equal to expected

        <b>Steps</b>
        1. Check _CountingAttr object returned by "selectable" function for default values of optional parameters
        2. Check _CountingAttr object returned by "selectable" function for specified values of optional parameters
        """

        def check_selectable(
            selectable_instance,
            expected_description: str = "Default selectable description",
            expected_warning: str = None,
            expected_editable: bool = True,
            expected_visible_in_ui: bool = True,
            expected_affects_outcome_of: ModelLifecycle = ModelLifecycle.NONE,
            expected_ui_rules: UIRules = NullUIRules(),
            expected_auto_hpo_state: AutoHPOState = AutoHPOState.NOT_POSSIBLE,
            expected_auto_hpo_value: SomeEnumSelectable = None,
        ):
            expected_metadata = {
                "default_value": SomeEnumSelectable.OPTION_C,
                "description": expected_description,
                "header": "selectable header",
                "warning": expected_warning,
                "editable": expected_editable,
                "visible_in_ui": expected_visible_in_ui,
                "affects_outcome_of": expected_affects_outcome_of,
                "ui_rules": expected_ui_rules,
                "type": ConfigElementType.SELECTABLE,
                "enum_name": "SomeEnumSelectable",
                "options": {
                    "TEST_NAME1": "test_name_1",
                    "TEST_2": "test_2_test",
                    "BOGUS_NAME": "bogus",
                    "OPTION_C": "option_c",
                },
                "auto_hpo_state": expected_auto_hpo_state,
                "auto_hpo_value": expected_auto_hpo_value,
            }
            assert isinstance(selectable_instance, _make._CountingAttr)
            assert selectable_instance._default == SomeEnumSelectable.OPTION_C
            assert selectable_instance.type == ConfigurableEnum
            assert selectable_instance.metadata == expected_metadata

        # Checking _CountingAttr object returned by "selectable" for default values of optional parameters
        default_value = SomeEnumSelectable.OPTION_C
        header = "selectable header"
        actual_selectable = selectable(default_value=default_value, header=header)
        check_selectable(selectable_instance=actual_selectable)  # type: ignore

        # Checking _CountingAttr object returned by "selectable" for specified values of optional parameters
        description = "selectable description"
        warning = "selectable warning"
        editable = False
        visible_in_ui = False
        affects_outcome_of = ModelLifecycle.TESTING
        ui_rules = self.ui_rules
        auto_hpo_state = AutoHPOState.POSSIBLE
        auto_hpo_value = SomeEnumSelectable.BOGUS_NAME

        actual_selectable = selectable(
            default_value=default_value,
            header=header,
            description=description,
            warning=warning,
            editable=editable,
            visible_in_ui=visible_in_ui,
            affects_outcome_of=affects_outcome_of,
            ui_rules=ui_rules,
            auto_hpo_value=auto_hpo_value,
            auto_hpo_state=auto_hpo_state,
        )
        check_selectable(
            selectable_instance=actual_selectable,  # type: ignore
            expected_description=description,
            expected_warning=warning,
            expected_editable=editable,
            expected_visible_in_ui=visible_in_ui,
            expected_affects_outcome_of=affects_outcome_of,
            expected_ui_rules=ui_rules,
            expected_auto_hpo_state=auto_hpo_state,
            expected_auto_hpo_value=auto_hpo_value,
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_string_attribute(self):
        """
        <b>Description:</b>
        Check "string_attribute" function

        <b>Input data:</b>
        "value" string

        <b>Expected results:</b>
        Test passes if _CountingAttr object returned by "string_attribute" function is equal to expected
        """
        value = "some string"
        actual_string = string_attribute(value=value)
        assert isinstance(actual_string, _make._CountingAttr)
        assert actual_string._default == value  # type: ignore
        assert actual_string.kw_only  # type: ignore
        assert actual_string.type == str  # type: ignore
        assert not actual_string._validator  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_boolean_attribute(self):
        """
        <b>Description:</b>
        Check "boolean_attribute" function

        <b>Input data:</b>
        "value" bool

        <b>Expected results:</b>
        Test passes if _CountingAttr object returned by "boolean_attribute" function is equal to expected
        """
        value = True
        actual_boolean = boolean_attribute(value=value)
        assert isinstance(actual_boolean, _make._CountingAttr)
        assert actual_boolean._default == value  # type: ignore
        assert actual_boolean.kw_only  # type: ignore
        assert actual_boolean.type == bool  # type: ignore
        assert not actual_boolean._validator  # type: ignore
