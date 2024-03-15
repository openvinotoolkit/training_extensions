# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from types import FunctionType

import pytest
from attr import fields

from otx.api.configuration import ConfigurableParameters
from otx.api.configuration.elements.parameter_group import ParameterGroup
from otx.api.configuration.elements.utils import (
    _convert_enum_selectable_value,
    _validate_and_convert_float,
    attr_enum_to_str_serializer,
    attr_strict_float_converter,
    attr_strict_float_on_setattr,
    attr_strict_int_validator,
    construct_attr_enum_selectable_converter,
    construct_attr_enum_selectable_onsetattr,
    construct_attr_selectable_validator,
    construct_attr_value_validator,
    convert_string_to_id,
)
from otx.api.configuration.enums.config_element_type import ElementCategory
from otx.api.entities.id import ID
from tests.unit.api.configuration.dummy_config import SomeEnumSelectable
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestUtilsFunctions:
    parameter_group = ParameterGroup(header="test header")
    attribute = fields(ConfigurableParameters)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_utils_attr_enum_to_str_serializer(self):
        """
        <b>Description:</b>
        Check "attr_enum_to_str_serializer" function

        <b>Input data:</b>
        "instance" Enum object, "attribute" Attribute object, "value" parameter

        <b>Expected results:</b>
        Test passes if value returned by "attr_enum_to_str_serializer" function is equal to expected

        <b>Steps</b>
        1. Check value returned by "attr_enum_to_str_serializer" function when "Enum" object is specified as "value"
        parameter
        2. Check value returned by "attr_enum_to_str_serializer" function when "str" object is specified as "value"
        parameter
        """
        # Checking value returned by "attr_enum_to_str_serializer" when "Enum" object is specified as "value"
        assert (
            attr_enum_to_str_serializer(
                instance=ElementCategory,
                attribute=ElementCategory.PRIMITIVES.name,
                value=ElementCategory.PRIMITIVES,
            )
            == "PRIMITIVES"
        )
        assert (
            attr_enum_to_str_serializer(
                instance=ElementCategory,
                attribute=ElementCategory.RULES.name,
                value=ElementCategory.RULES,
            )
            == "RULES"
        )
        # Checking value returned by "attr_enum_to_str_serializer" when "str" object is specified as "value"
        assert (
            attr_enum_to_str_serializer(
                instance=ElementCategory,
                attribute=self.attribute.id,
                value="non enum string",
            )
            == "non enum string"
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_convert_enum_selectable_value(self):
        """
        <b>Description:</b>
        Check "_convert_enum_selectable_value" function

        <b>Input data:</b>
        "value" parameter, "enum_class" ConfigurableEnum object

        <b>Expected results:</b>
        Test passes if ConfigurableEnum class element returned by "_convert_enum_selectable_value" function is
        equal to expected

        <b>Steps</b>
        1. Check ConfigurableEnum class element returned by "_convert_enum_selectable_value" function when string
        is specified as "value" parameter
        2. Check ConfigurableEnum class element returned by "_convert_enum_selectable_value" function when
        ConfigurableEnum class element is specified as "value" parameter
        3. Check that ValueError exception is raised by "_convert_enum_selectable_value" function when unexpected string
        is specified as "value" parameter
        """
        # Checking ConfigurableEnum element returned by "_convert_enum_selectable_value" when string is specified as
        # "value"
        assert (
            _convert_enum_selectable_value(value="test_2_test", enum_class=SomeEnumSelectable)
            == SomeEnumSelectable.TEST_2
        )
        # Checking ConfigurableEnum element returned by "_convert_enum_selectable_value" when ConfigurableEnum element
        # is specified as "value"
        assert (
            _convert_enum_selectable_value(value=SomeEnumSelectable.OPTION_C, enum_class=SomeEnumSelectable)
            == SomeEnumSelectable.OPTION_C
        )
        # Checking that ValueError exception is raised by "_convert_enum_selectable_value" when unexpected string is
        # specified as "value"
        with pytest.raises(ValueError):
            _convert_enum_selectable_value(value="some string", enum_class=SomeEnumSelectable)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_construct_attr_enum_selectable_converter(self):
        """
        <b>Description:</b>
        Check "construct_attr_enum_selectable_converter" function

        <b>Input data:</b>
        "default_value" ConfigurableEnum element

        <b>Expected results:</b>
        Test passes if function returned by "construct_attr_enum_selectable_converter" function is equal to expected
        """
        converter = construct_attr_enum_selectable_converter(default_value=SomeEnumSelectable.TEST_NAME1)
        assert isinstance(converter, FunctionType)
        assert converter(SomeEnumSelectable.BOGUS_NAME) == SomeEnumSelectable.BOGUS_NAME
        assert converter("test_2_test") == SomeEnumSelectable.TEST_2

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_construct_attr_enum_selectable_onsetattr(self):
        """
        <b>Description:</b>
        Check "construct_attr_enum_selectable_onsetattr" function

        <b>Input data:</b>
        "default_value" ConfigurableEnum element, "parameter_group" ParameterGroup object, "attribute" Attribute object,
        "value" parameter

        <b>Expected results:</b>
        Test passes if function returned by "construct_attr_enum_selectable_onsetattr" function is equal to expected
        """
        on_set_attr = construct_attr_enum_selectable_onsetattr(default_value=SomeEnumSelectable.TEST_2)
        assert isinstance(on_set_attr, FunctionType)
        assert (
            on_set_attr(
                self.parameter_group,
                SomeEnumSelectable.TEST_2.value,
                SomeEnumSelectable.TEST_2,
            )
            == SomeEnumSelectable.TEST_2
        )
        assert (
            on_set_attr(self.parameter_group, SomeEnumSelectable.OPTION_C.value, "option_c")
            == SomeEnumSelectable.OPTION_C
        )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_construct_attr_value_validator(self):
        """
        <b>Description:</b>
        Check "construct_attr_value_validator" function

        <b>Input data:</b>
        "min_value" and "max_value" parameters, "parameter_group" ParameterGroup object, "attribute" Attribute object,
        "value" parameter

        <b>Expected results:</b>
        Test passes if validator returned by "construct_attr_value_validator" function is equal to expected

        <b>Steps</b>
        1. Check that ValueError exception is not raised by validator returned by "construct_attr_value_validator"
        function for values within specified bounds
        2. Check that ValueError exception is raised by validator returned by "construct_attr_value_validator" function
        for values out of specified bounds
        """
        attr_value_validator = construct_attr_value_validator(min_value=1, max_value=4)
        # Checking that ValueError exception is not raised by validator returned by "construct_attr_value_validator"
        # for values within specified bounds
        for value_within in range(1, 5):
            attr_value_validator(self.parameter_group, self.attribute.id, value_within)
        # Checking that ValueError exception is raised by validator returned by "construct_attr_value_validator" for
        # values out of specified bounds
        for out_of_bounds_value in [0, 5]:
            with pytest.raises(ValueError):
                attr_value_validator(self.parameter_group, self.attribute.id, out_of_bounds_value)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_construct_attr_selectable_validator(self):
        """
        <b>Description:</b>
        Check "construct_attr_selectable_validator" function

        <b>Input data:</b>
        "options" list

        <b>Expected results:</b>
        Test passes if validator returned by "construct_attr_selectable_validator" function is equal to expected

        <b>Steps</b>
        1. Check that ValueError exception is not raised by validator returned by "construct_attr_selectable_validator"
        function for values included in "options" list
        2. Check that ValueError exception is raised by validator returned by "construct_attr_selectable_validator"
        function for values not included in "options" list
        """
        attr_selectable_validator = construct_attr_selectable_validator(options=["str_option", 2])
        # Checking that ValueError exception is not raised by validator returned by
        # "construct_attr_selectable_validator" for values included in "options"
        for value_within in ["str_option", 2]:
            attr_selectable_validator(self.parameter_group, self.attribute.id, value_within)
        # Checking that ValueError exception is raised by validator returned by "construct_attr_selectable_validator"
        # for values not included in "options"
        for out_of_bounds_value in ["other_str_option", 3]:
            with pytest.raises(ValueError):
                attr_selectable_validator(self.parameter_group, self.attribute.id, out_of_bounds_value)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_convert_string_to_id(self):
        """
        <b>Description:</b>
        Check "convert_string_to_id" function

        <b>Input data:</b>
        "id_string" string or ID object

        <b>Expected results:</b>
        Test passes if ID object returned by "convert_string_to_id" function is equal to expected

        <b>Steps</b>
        1. Check ID object returned by "convert_string_to_id" function for string "id_string" parameter
        2. Check ID object returned by "convert_string_to_id" function for ID "id_string" parameter
        3. Check ID object returned by "convert_string_to_id" function for "id_string" parameter equal to None
        4. Check ID object returned by "convert_string_to_id" function for int "id_string" parameter
        """
        # Checking ID returned by "convert_string_to_id" for string "id_string"
        assert convert_string_to_id("some_id") == ID("some_id")
        # Checking ID returned by "convert_string_to_id" for ID "id_string"
        assert convert_string_to_id(ID("id_string")) == ID("id_string")
        # Checking ID returned by "convert_string_to_id" for "id_string" equal to None
        assert convert_string_to_id(None) == ID()
        # Checking ID returned by "convert_string_to_id" for int "id_string"
        assert convert_string_to_id(4) == 4  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_attr_strict_int_validator(self):
        """
        <b>Description:</b>
        Check "attr_strict_int_validator" function

        <b>Input data:</b>
        "parameter_group" ParameterGroup object, "attribute" Attribute object, "value" parameter

        <b>Expected results:</b>
        Test passes if "attr_strict_int_validator" function raises TypeError exception for non-int "value"
        parameter

        <b>Steps</b>
        1. Check that "attr_strict_int_validator" function not raises TypeError exception for int "value" parameter
        2. Check that "attr_strict_int_validator" function raises TypeError exception for bool "value" parameter
        3. Check that "attr_strict_int_validator" function raises TypeError exception for string "value" parameter
        """
        # Checking that "attr_strict_int_validator" not raises TypeError exception for int "value"
        attr_strict_int_validator(instance=self.parameter_group, attribute=self.attribute.id, value=1)
        # Checking that "attr_strict_int_validator" raises TypeError exception for bool "value"
        with pytest.raises(TypeError):
            attr_strict_int_validator(instance=self.parameter_group, attribute=self.attribute.id, value=True)
        # Checking that "attr_strict_int_validator" raises TypeError exception for string "value"
        with pytest.raises(TypeError):
            attr_strict_int_validator(
                instance=self.parameter_group,
                attribute=self.attribute.id,
                value="some string",  # type: ignore
            )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_validate_and_convert_float(self):
        """
        <b>Description:</b>
        Check "_validate_and_convert_float" function

        <b>Input data:</b>
        "value" parameter

        <b>Expected results:</b>
        Test passes if value returned by "_validate_and_convert_float" function is equal to expected

        <b>Steps</b>
        1. Check value returned by "_validate_and_convert_float" function for float "value" parameter
        2. Check value returned by "_validate_and_convert_float" function for int "value" parameter
        3. Check value returned by "_validate_and_convert_float" function for bool "value" parameter
        4. Check value returned by "_validate_and_convert_float" function for str "value" parameter
        """
        # Checking value returned by "_validate_and_convert_float" for float "value"
        assert _validate_and_convert_float(value=1.3) == 1.3
        # Checking value returned by "_validate_and_convert_float" for int "value"
        converted_value = _validate_and_convert_float(value=2)
        assert isinstance(converted_value, float)
        assert converted_value == float(2)
        # Checking value returned by "_validate_and_convert_float" for bool "value"
        assert not _validate_and_convert_float(value=True)
        assert not _validate_and_convert_float(value=False)
        # Checking value returned by "_validate_and_convert_float" for str "value"
        assert not _validate_and_convert_float(value="some string")  # type: ignore

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_attr_strict_float_on_setattr(self):
        """
        <b>Description:</b>
        Check "attr_strict_float_on_setattr" function

        <b>Input data:</b>
        "parameter_group" ParameterGroup object, "attribute" Attribute object, "value" parameter

        <b>Expected results:</b>
        Test passes if value returned by "attr_strict_float_on_setattr" function is equal to expected

        <b>Steps</b>
        1. Check value returned by "attr_strict_float_on_setattr" function for float "value" parameter
        2. Check value returned by "attr_strict_float_on_setattr" function for int "value" parameter
        3. Check that "attr_strict_float_on_setattr" function raises TypeError exception for bool "value" parameter
        4. Check that "attr_strict_float_on_setattr" function raises TypeError exception for str "value" parameter
        """
        # Checking value returned by "attr_strict_float_on_setattr" for float "value"
        assert (
            attr_strict_float_on_setattr(instance=self.parameter_group, attribute=self.attribute.id, value=10.7) == 10.7
        )
        # Checking value returned by "attr_strict_float_on_setattr" for int "value"
        converted_value = attr_strict_float_on_setattr(
            instance=self.parameter_group, attribute=self.attribute.id, value=2
        )
        assert isinstance(converted_value, float)
        assert converted_value == float(2)
        # Checking that "attr_strict_float_on_setattr" raises TypeError exception for bool "value"
        with pytest.raises(TypeError):
            attr_strict_float_on_setattr(instance=self.parameter_group, attribute=self.attribute.id, value=True)
        # Checking that "attr_strict_float_on_setattr" raises TypeError exception for str "value"
        with pytest.raises(TypeError):
            attr_strict_float_on_setattr(
                instance=self.parameter_group,
                attribute=self.attribute.id,
                value="some string",  # type: ignore
            )

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_attr_strict_float_converter(self):
        """
        <b>Description:</b>
        Check "attr_strict_float_converter" function

        <b>Input data:</b>
        "value" parameter

        <b>Expected results:</b>
        Test passes if value returned by "attr_strict_float_converter" function is equal to expected

        <b>Steps</b>
        1. Check value returned by "attr_strict_float_converter" function for float "value" parameter
        2. Check value returned by "attr_strict_float_converter" function for int "value" parameter
        3. Check value returned by "attr_strict_float_converter" function for bool or str "value" parameter
        """
        # Checking value returned by "attr_strict_float_converter" for float "value"
        assert attr_strict_float_converter(value=20.1) == 20.1
        # Checking value returned by "attr_strict_float_converter" for int "value"
        converted_value = attr_strict_float_converter(value=0)
        assert isinstance(converted_value, float)
        assert converted_value == float(0)
        # Checking that "attr_strict_float_converter" raises TypeError for bool or str "value"
        for non_bool_value in [True, False, "some string"]:
            with pytest.raises(TypeError):
                attr_strict_float_converter(non_bool_value)  # type: ignore
