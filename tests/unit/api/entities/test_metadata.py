"""This module tests classes related to metadata"""

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


import re

import pytest

from otx.api.entities.metadata import (
    FloatMetadata,
    FloatType,
    IMetadata,
    MetadataItemEntity,
)
from otx.api.entities.model import ModelEntity
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIMetadata:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_imetadata(self):
        """
        <b>Description:</b>
        To test IMetadata class

        <b>Input data:</b>
        Initialized instance of IMetadata class

        <b>Expected results:</b>
        1. Initialized instance is instance of the class
        2. Default value of field has expected value:  "typing.Union[str, NoneType]"
        3. Changed fields value has expected value: "String"

        <b>Steps</b>
        1. Create IMetadata
        2. Check default value of class field
        3. Change value of the field
        """

        test_instance = IMetadata()
        assert isinstance(test_instance, IMetadata)
        assert str(test_instance.name) in ["typing.Union[str, NoneType]", "typing.Optional[str]"]

        test_instance.name = "String"
        assert test_instance.name == "String"


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestFloatType:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_float_type_members(self):
        """
        <b>Description:</b>
        To test FloatType enumeration members

        <b>Input data:</b>
        Initialized instance of FloatType enum class

        <b>Expected results:</b>
        1. Enum members return correct values:
            FLOAT = 1
            EMBEDDING_VALUE = 2
            ACTIVE_SCORE = 3
        2. In case incorrect member it raises AttributeError exception
        3. In case incorrect member value it raises ValueError exception

        <b>Steps</b>
        0. Create FloatType
        1. Check members
        2. Check incorrect member
        3. Check incorrect member value
        """

        test_instance = FloatType

        for i in range(1, 4):
            assert test_instance(i) in list(FloatType)

        with pytest.raises(AttributeError):
            test_instance.WRONG

        with pytest.raises(ValueError):
            test_instance(6)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_float_type_magic_str(self):
        """
        <b>Description:</b>
        To test FloatType __str__ method

        <b>Input data:</b>
        Initialized instance of FloatType enum

        <b>Expected results:</b>
        1. __str__ returns correct string for every enum member
        2. In case incorrect member it raises AttributeError exception
        3. In case incorrect member value it raises ValueError exception

        <b>Steps</b>
        0. Create FloatType
        1. Check returning value of __str__ method
        2. Try incorrect field name
        3. Try incorrect value name
        """
        test_instance = FloatType
        magic_str_list = [str(i) for i in list(FloatType)]

        for i in range(1, 4):
            assert str(test_instance(i)) in magic_str_list

        with pytest.raises(AttributeError):
            str(test_instance.WRONG)

        with pytest.raises(ValueError):
            str(test_instance(6))


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestMetadataItemEntity:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_metadata_item_entity(self):
        """
        <b>Description:</b>
        To test MetadataItemEntity class

        <b>Input data:</b>
        Initialized instances of IMetadata, ModelEntity and MetadataItemEntity classes

        <b>Expected results:</b>
        1. Initialized instances
        2. Default values of fields are None
        3. repr method returns expected value "NullMetadata(None, None)" after initiation for both instances
        4. Instance test_instance1 fields values changed and have expected values:
            name == String1
            value == 0
        5. repr method of test_instance1 returns expected value "NullMetadata(String1, 0)"
        6. == method behavior is expected

        <b>Steps</b>
        1. Create class instances test_instance0 and test_instance1
        2. Perform checks of its field default values after initiations
        3. Perform checks of repr method returns expected values after initiations
        4. Change fields value of test_instance1
        5. Perform checks of repr method returns expected values after changes
        6. Check that test_instance0 == test_instance1
        """
        i_metadata = IMetadata()
        i_metadata.name = "default_i_metadata"
        test_data0 = test_data1 = i_metadata.name
        i_metadata.name = "i_metadata"
        test_data2 = i_metadata.name
        test_model0 = test_model1 = ModelEntity(train_dataset="default_dataset", configuration="default_config")
        test_instance0 = MetadataItemEntity(test_data0, test_model0)
        test_instance1 = MetadataItemEntity(test_data1, test_model1)
        test_instance2 = MetadataItemEntity(test_data2, test_model1)
        assert test_instance0 == test_instance1 != test_instance2
        __repr = repr(test_instance0)
        repr_pattern = (
            r"MetadataItemEntity\(model=\<otx.api.entities.model.ModelEntity object at" r" 0x[a-fA-F0-9]{10,32}\>\)"
        )
        assert re.match(repr_pattern, __repr)


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestFloatMetadata:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_float_metadata(self):
        """
        <b>Description:</b>
        To test FloatMetadata class

        <b>Input data:</b>
        Initialized instance of FloatMetadata

        <b>Expected results:</b>
        1. It raises TypeError in case attempt of initiation with wrong parameters numbers
        2. Fields of instances are with correct values
        3. repr method returns correct strings then used against each instance
        4. '==' method works as expected

        <b>Steps</b>
        1. Attempt to initiate class instance with wrong parameters numbers
        2. Initiate three class instances:
            two of them with similar set of init values, third one with different one.
        3. Check repr method
        4. Check __eq__ method
        """
        with pytest.raises(TypeError):
            FloatMetadata()

        with pytest.raises(TypeError):
            FloatMetadata("only name")

        test_inst0 = FloatMetadata(name="Instance0", value=42)
        assert test_inst0.name == "Instance0"
        assert test_inst0.value == 42
        assert repr(test_inst0.float_type) == "<FloatType.FLOAT: 1>"
        assert repr(test_inst0) == "FloatMetadata(Instance0, 42, FLOAT)"

        test_inst1 = FloatMetadata(name="Instance1", value=42.0)
        assert test_inst1.name == "Instance1"
        assert test_inst1.value == 42.0
        assert repr(test_inst1.float_type) == "<FloatType.FLOAT: 1>"
        assert repr(test_inst1) == "FloatMetadata(Instance1, 42.0, FLOAT)"

        test_inst2 = FloatMetadata(name="Instance0", value=42)
        assert test_inst2.name == "Instance0"
        assert test_inst2.value == 42
        assert repr(test_inst2.float_type) == "<FloatType.FLOAT: 1>"
        assert repr(test_inst2) == "FloatMetadata(Instance0, 42, FLOAT)"

        assert test_inst0 == test_inst2 != test_inst1
