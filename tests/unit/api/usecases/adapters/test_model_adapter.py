# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.api.usecases.adapters.model_adapter import (
    ExportableCodeAdapter,
    IDataSource,
    ModelAdapter,
)
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestIDataSource:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_i_data_source_data(self):
        """
        <b>Description:</b>
        Check IDataSource class "data" property

        <b>Input data:</b>
        IDataSource object

        <b>Expected results:</b>
        Test passes if IDataSource object "data" property raises NotImplementedError exception
        """
        with pytest.raises(NotImplementedError):
            IDataSource().data()


class DummyDataSource(IDataSource):
    def __init__(self, data: str):
        self._data = data

    @property
    def data(self):
        return self._data


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestModelAdapter:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_adapter_initialization(self):
        """
        <b>Description:</b>
        Check ModelAdapter object initialization

        <b>Input data:</b>
        ModelAdapter object initialized with "data_source" IDataSource or bytes parameter

        <b>Expected results:</b>
        Test passes if properties of initialized ModelAdapter object are equal to expected

        <b>Steps</b>
        1. Check attributes of ModelAdapter object initialized with IDataSource "data_source" parameter
        2. Check attributes of ModelAdapter object initialized with bytes "data_source" parameter
        3. Check that ValueError exception is raised when initializing ModelAdapter object with "data_source" parameter
        type not equal to bytes or IDataSource
        """
        # Checking properties of "ModelAdapter" initialized with IDataSource "data_source"
        data = "some data"
        data_source = DummyDataSource(data=data)
        model_adapter = ModelAdapter(data_source=data_source)
        assert model_adapter.data_source == data_source
        assert model_adapter.from_file_storage
        assert model_adapter.data == data
        # Checking properties of "ModelAdapter" initialized with bytes "data_source"
        data_source = b"binaryrepo://localhost/repo/data_source/1"
        model_adapter = ModelAdapter(data_source=data_source)
        assert model_adapter.data_source == data_source
        assert not model_adapter.from_file_storage
        assert model_adapter.data == data_source
        # Checking that ValueError exception is raised when initializing ModelAdapter with "data_source" type not equal
        # to bytes or IDataSource
        model_adapter = ModelAdapter(data_source=1)  # type: ignore
        with pytest.raises(ValueError):
            model_adapter.data()

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_model_adapter_data_source_setter(self):
        """
        <b>Description:</b>
        Check ModelAdapter "data_source" property setter

        <b>Input data:</b>
        ModelAdapter object initialized with "data_source" parameter

        <b>Expected results:</b>
        Test passes if properties of ModelAdapter are equal to expected after manual setting "data_source" property

        <b>Steps</b>
        1. Check properties of ModelAdapter object after manual setting "data_source" property to other IDataSource
        object
        2. Check properties of ModelAdapter object after manual setting "data_source" property to bytes object
        3. Check properties of ModelAdapter object after manual setting "data_source" property to other bytes object
        4. Check properties of ModelAdapter object after manual setting "data_source" property to IDataSource object
        """
        model_adapter = ModelAdapter(data_source=DummyDataSource(data="some data"))
        # Checking properties of ModelAdapter after manual setting "data_source" to other IDataSource
        other_data = "other data"
        other_data_source = DummyDataSource(data=other_data)
        model_adapter.data_source = other_data_source
        assert model_adapter.data_source == other_data_source
        assert model_adapter.data == other_data
        assert model_adapter.from_file_storage
        # Checking properties of ModelAdapter after manual setting "data_source" to bytes
        bytes_data_source = b"binaryrepo://localhost/repo/data_source/1"
        model_adapter.data_source = bytes_data_source
        assert model_adapter.data_source == bytes_data_source
        assert model_adapter.data == bytes_data_source
        assert not model_adapter.from_file_storage
        # Checking properties of ModelAdapter after manual setting "data_source" to other bytes
        other_bytes_data_source = b"binaryrepo://localhost/repo/data_source/2"
        model_adapter.data_source = b"binaryrepo://localhost/repo/data_source/2"
        assert model_adapter.data_source == other_bytes_data_source
        assert model_adapter.data == other_bytes_data_source
        assert not model_adapter.from_file_storage
        # Checking properties of ModelAdapter after manual setting "data_source" to IDataSource
        model_adapter.data_source = other_data_source
        assert model_adapter.data_source == other_data_source
        assert model_adapter.data == other_data
        assert model_adapter.from_file_storage


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestExportableCodeAdapter:
    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_exportable_code_adapter_initialization(self):
        """
        <b>Description:</b>
        Check ExportableCodeAdapter object initialization

        <b>Input data:</b>
        ExportableCodeAdapter object initialized with "data_source" IDataSource or bytes parameter

        <b>Expected results:</b>
        Test passes if properties of initialized ExportableCodeAdapter object are equal to expected

        <b>Steps</b>
        1. Check attributes of ExportableCodeAdapter object initialized with IDataSource "data_source" parameter
        2. Check attributes of ExportableCodeAdapter object initialized with bytes "data_source" parameter
        3. Check that ValueError exception is raised when initializing ExportableCodeAdapter object with "data_source"
        parameter type not equal to bytes or IDataSource
        """
        # Checking properties of "ExportableCodeAdapter" initialized with IDataSource "data_source"
        data = "some_data"
        data_source = DummyDataSource(data=data)
        exportable_code_adapter = ExportableCodeAdapter(data_source=data_source)
        assert exportable_code_adapter.data_source == data_source
        assert exportable_code_adapter.from_file_storage
        assert exportable_code_adapter.data == data
        # Checking properties after setting "data_source" to bytes
        bytes_data_source = b"binaryrepo://localhost/repo/data_source/1"
        exportable_code_adapter.data_source = bytes_data_source
        assert exportable_code_adapter.data_source == bytes_data_source
        assert not exportable_code_adapter.from_file_storage
        assert exportable_code_adapter.data == bytes_data_source
        # Checking properties of "ExportableCodeAdapter" initialized with bytes "data_source"
        exportable_code_adapter = ExportableCodeAdapter(data_source=bytes_data_source)
        assert exportable_code_adapter.data_source == bytes_data_source
        assert not exportable_code_adapter.from_file_storage
        assert exportable_code_adapter.data == bytes_data_source
        # Checking properties after setting "data_source" to IDataSource
        exportable_code_adapter.data_source = data_source
        assert exportable_code_adapter.data_source == data_source
        assert exportable_code_adapter.from_file_storage
        assert exportable_code_adapter.data == data
        # Checking that ValueError exception is raised when initializing ExportableCodeAdapter with "data_source" type
        # not equal to bytes or IDataSource
        exportable_code_adapter = ExportableCodeAdapter(data_source=1)  # type: ignore
        with pytest.raises(ValueError):
            exportable_code_adapter.data()
