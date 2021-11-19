"""This module tests classes related to metadata"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.

import pytest
import re

from ote_sdk.entities.metadata import FloatType
from ote_sdk.entities.metadata import MetadataItemEntity
from ote_sdk.entities.metadata import FloatMetadata
from ote_sdk.entities.metadata import IMetadata
from ote_sdk.entities.model import ModelEntity
from ote_sdk.tests.constants.ote_sdk_components import OteSdkComponent
from ote_sdk.tests.constants.requirements import Requirements


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestIMetadata:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_imetadata(self):
        """
            <b>Description:</b>
            To test IMetadata class

            <b>Input data:</b>
            Initiated instance of IMetadata class

            <b>Expected results:</b>
            1. Initiated instance are instance of the class
            2. Default value of field has expected value:  "typing.Union[str, NoneType]
            3. Changed fields value has expected value: "String"

            <b>Steps</b>
            1. Initiate IMetadata class instance
            2. Check default value of class field
            3. Change value of the field
        """

        test_instance = IMetadata()
        assert isinstance(test_instance, IMetadata)
        assert str(test_instance.name) == "typing.Union[str, NoneType]"

        test_instance.name = "String"
        assert test_instance.name == "String"


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestFloatType:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_float_type_members(self):
        """
                <b>Description:</b>
                To test FloatType enumeration members

                <b>Input data:</b>
                Initiated instance of FloatType enum class

                <b>Expected results:</b>
                1. Enum members return correct values:
                    FLOAT = 1
                    EMBEDDING_VALUE = 2
                    ACTIVE_SCORE = 3
                2. In case incorrect member it raises AttributeError exception
                3. In case incorrect member value it raises ValueError exception

                <b>Steps</b>
                0. Initiate enum instance
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
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_float_type_magic_str(self):
        """
                <b>Description:</b>
                To test FloatType __str__ method

                <b>Input data:</b>
                Initiated instance of FloatType enum

                <b>Expected results:</b>
                1. __str__ return correct string for every enum member
                2. In case incorrect member it raises AttributeError exception
                3. In case incorrect member value it raises ValueError exception

                <b>Steps</b>
                0. Initiate enum instance
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


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestMetadataItemEntity:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_metadata_item_entity(self):
        """
        <b>Description:</b>
        To test MetadataItemEntity class

        <b>Input data:</b>
        Initiated instances of IMetadata, ModelEntity and MetadataItemEntity classes

        <b>Expected results:</b>
        1. Initiated instances
        2. Default values of fields are None
        3. repr method returns expected value "NullMetadata(None, None)" after initiation for both instances
        4. Instance test_instance1 fields values changed and have expected values:
            name == String1
            value == 0
        5. repr method of test_instance1 returns expected value "NullMetadata(String1, 0)"
        6. == method behavior is expected

        <b>Steps</b>
        1. Initiate class instances test_instance0 and test_instance1
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
        repr_pattern = r'MetadataItemEntity\(model=\<ote_sdk.entities.model.ModelEntity object at' \
                       r' 0x[a-fA-F0-9]{10,32}\>\, data\=default_i_metadata\)'
        assert re.match(repr_pattern, __repr)


@pytest.mark.components(OteSdkComponent.OTE_SDK)
class TestFloatMetadata:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_float_metadata(self):
        """
            <b>Description:</b>
            To test FloatMetadata class

            <b>Input data:</b>
            Initiated instance of FloatMetadata

            <b>Expected results:</b>
            1. It raises TypeError in case attempt of initiation with wrong parameters numbers
            2. Fields of instances initiated with correct values
            3. repr method returns correct strings then used against each instance
            4. '==' method works is expected

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

        test_inst1 = FloatMetadata(name="Instance1", value=42.)
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
