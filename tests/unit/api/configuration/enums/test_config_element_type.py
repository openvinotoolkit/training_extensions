# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.api.configuration.enums.config_element_type import (
    ConfigElementType,
    ElementCategory,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestElementCategory:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_element_category(self):
        """
        <b>Description:</b>
        Check "ElementCategory" Enum class elements

        <b>Expected results:</b>
        Test passes if "ElementCategory" Enum class length is equal to expected value and its elements have expected
        sequence numbers, "name" attributes and values returned by __str__ method
        """
        assert len(ElementCategory) == 3
        assert ElementCategory.PRIMITIVES.value == 1
        assert ElementCategory.PRIMITIVES.name == "PRIMITIVES"
        assert str(ElementCategory.PRIMITIVES) == "PRIMITIVES"
        assert ElementCategory.GROUPS.value == 2
        assert ElementCategory.GROUPS.name == "GROUPS"
        assert str(ElementCategory.GROUPS) == "GROUPS"
        assert ElementCategory.RULES.value == 3
        assert ElementCategory.RULES.name == "RULES"
        assert str(ElementCategory.RULES) == "RULES"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestConfigElementType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_element_category(self):
        """
        <b>Description:</b>
        Check "ConfigElementType" Enum class elements

        <b>Expected results:</b>
        Test passes if "ConfigElementType" Enum class length is equal to expected value and its elements have expected
        sequence numbers, "names" attributes, "category" properties and values returned by and __str__ method
        """
        assert len(ConfigElementType) == 9
        # Checking "INTEGER" element
        assert ConfigElementType.INTEGER.category == ElementCategory.PRIMITIVES
        assert ConfigElementType.INTEGER.name == "INTEGER"
        assert ConfigElementType.INTEGER.value == 0
        assert str(ConfigElementType.INTEGER) == "INTEGER"
        # Checking "FLOAT" element
        assert ConfigElementType.FLOAT.category == ElementCategory.PRIMITIVES
        assert ConfigElementType.FLOAT.name == "FLOAT"
        assert ConfigElementType.FLOAT.value == 1
        assert str(ConfigElementType.FLOAT) == "FLOAT"
        # Checking "BOOLEAN" element
        assert ConfigElementType.BOOLEAN.category == ElementCategory.PRIMITIVES
        assert ConfigElementType.BOOLEAN.name == "BOOLEAN"
        assert ConfigElementType.BOOLEAN.value == 2
        assert str(ConfigElementType.BOOLEAN) == "BOOLEAN"
        # Checking "FLOAT_SELECTABLE" element
        assert ConfigElementType.FLOAT_SELECTABLE.category == ElementCategory.PRIMITIVES
        assert ConfigElementType.FLOAT_SELECTABLE.name == "FLOAT_SELECTABLE"
        assert ConfigElementType.FLOAT_SELECTABLE.value == 3
        assert str(ConfigElementType.FLOAT_SELECTABLE) == "FLOAT_SELECTABLE"
        # Checking "SELECTABLE" element
        assert ConfigElementType.SELECTABLE.category == ElementCategory.PRIMITIVES
        assert ConfigElementType.SELECTABLE.name == "SELECTABLE"
        assert ConfigElementType.SELECTABLE.value == 4
        assert str(ConfigElementType.SELECTABLE) == "SELECTABLE"
        # Checking "PARAMETER_GROUP" element
        assert ConfigElementType.PARAMETER_GROUP.category == ElementCategory.GROUPS
        assert ConfigElementType.PARAMETER_GROUP.name == "PARAMETER_GROUP"
        assert ConfigElementType.PARAMETER_GROUP.value == 5
        assert str(ConfigElementType.PARAMETER_GROUP) == "PARAMETER_GROUP"
        # Checking "CONFIGURABLE_PARAMETERS" element
        assert ConfigElementType.CONFIGURABLE_PARAMETERS.category == ElementCategory.GROUPS
        assert ConfigElementType.CONFIGURABLE_PARAMETERS.name == "CONFIGURABLE_PARAMETERS"
        assert ConfigElementType.CONFIGURABLE_PARAMETERS.value == 6
        assert str(ConfigElementType.CONFIGURABLE_PARAMETERS) == "CONFIGURABLE_PARAMETERS"
        # Checking "RULE" element
        assert ConfigElementType.RULE.category == ElementCategory.RULES
        assert ConfigElementType.RULE.name == "RULE"
        assert ConfigElementType.RULE.value == 7
        assert str(ConfigElementType.RULE) == "RULE"
        # Checking "RULE" element
        assert ConfigElementType.UI_RULES.category == ElementCategory.RULES
        assert ConfigElementType.UI_RULES.name == "UI_RULES"
        assert ConfigElementType.UI_RULES.value == 8
        assert str(ConfigElementType.UI_RULES) == "UI_RULES"
