# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from ote_sdk.configuration.enums.config_element_type import ConfigElementType
from ote_sdk.configuration.enums.utils import get_enum_names
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestMetadataKeys:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_get_enum_names(self):
        """
        <b>Description:</b>
        Check "get_enum_names" function

        <b>Input data:</b>
        Enum object

        <b>Expected results:</b>
        Test passes if list returned by "get_enum_names" function is equal to expected
        """
        assert get_enum_names(ConfigElementType) == [
            "INTEGER",
            "FLOAT",
            "BOOLEAN",
            "FLOAT_SELECTABLE",
            "SELECTABLE",
            "PARAMETER_GROUP",
            "CONFIGURABLE_PARAMETERS",
            "RULE",
            "UI_RULES",
        ]
