# Copyright (C) 2021-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest

from otx.api.entities.tensor import TensorEntity
from tests.unit.api.constants.components import OtxSdkComponent
from tests.unit.api.constants.requirements import Requirements

RANDOM_NUMPY = np.random.randint(low=0, high=255, size=(16, 32, 3))


@pytest.mark.components(OtxSdkComponent.OTX_API)
class TestTensorEntity:
    @staticmethod
    def tensor_params():
        return {"name": "Test Tensor", "numpy": RANDOM_NUMPY}

    def tensor(self):
        return TensorEntity(**self.tensor_params())

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_tensor_initialization(self):
        """
        <b>Description:</b>
        Check TensorEntity class object initialization

        <b>Input data:</b>
        TensorEntity class object with specified "name" and "numpy" parameters

        <b>Expected results:</b>
        Test passes if attributes of initialized TensorEntity class object are equal to expected
        """
        tensor_params = self.tensor_params()
        tensor = TensorEntity(**tensor_params)
        assert tensor.name == tensor_params.get("name")
        assert np.array_equal(tensor.numpy, tensor_params.get("numpy"))

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_tensor_shape(self):
        """
        <b>Description:</b>
        Check TensorEntity class object "shape" property

        <b>Input data:</b>
        TensorEntity class object with specified "name" and "numpy" parameters

        <b>Expected results:</b>
        Test passes if value returned by "shape" property is equal to expected

        <b>Steps</b>
        1. Check value returned by "shape" property for initialized TensorEntity object
        2. Manually set new value of "numpy" property and check re-check "numpy" and "shape" properties
        """
        # Checking values returned by "shape" property for initialized TensorEntity object
        tensor = self.tensor()
        assert tensor.shape == (16, 32, 3)
        # Manually setting new value of "numpy" property and re-checking "numpy and "shape" properties
        new_numpy = np.random.uniform(low=0.0, high=255.0, size=(8, 16, 3))
        tensor.numpy = new_numpy
        assert np.array_equal(tensor.numpy, new_numpy)
        assert tensor.shape == (8, 16, 3)

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_tensor_eq(self):
        """
        <b>Description:</b>
        Check TensorEntity class object __eq__ method

        <b>Input data:</b>
        TensorEntity class objects with specified "name" and "numpy" parameters

        <b>Expected results:</b>
        Test passes if value returned by __eq__ method is equal to expected

        <b>Steps</b>
        1. Check value returned by __eq__ method for comparing equal TensorEntity objects
        2. Check value returned by __eq__ method for comparing TensorEntity objects with unequal "name" parameters:
        expected equality
        3. Check value returned by __eq__ method for comparing TensorEntity objects with unequal "numpy" parameters -
        expected inequality
        4. Check value returned by __eq__ method for comparing TensorEntity with different type object
        """
        initialization_params = self.tensor_params()
        tensor = TensorEntity(**initialization_params)
        # Comparing equal TensorEntity objects
        equal_tensor = TensorEntity(**initialization_params)
        assert tensor == equal_tensor
        # Comparing TensorEntity objects with unequal "name" parameter, expected equality
        unequal_params = dict(initialization_params)
        unequal_params["name"] = "Unequal_name"
        equal_tensor = TensorEntity(**unequal_params)
        assert tensor == equal_tensor
        # Comparing TensorEntity objects with unequal "numpy" parameter, expected inequality
        unequal_params = dict(initialization_params)
        unequal_params["numpy"] = np.random.uniform(low=0.0, high=255.0, size=(1, 2, 3))
        unequal_tensor = TensorEntity(**unequal_params)
        assert tensor != unequal_tensor
        # Comparing TensorEntity with different type object
        assert tensor != str

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_tensor_str(self):
        """
        <b>Description:</b>
        Check TensorEntity class object __str__ method

        <b>Input data:</b>
        TensorEntity class object with specified "name" and "numpy" parameters

        <b>Expected results:</b>
        Test passes if value returned by __str__ method is equal to expected
        """
        tensor = self.tensor()
        assert str(tensor) == "TensorEntity(name=Test Tensor, shape=(16, 32, 3))"

    @pytest.mark.priority_medium
    @pytest.mark.unit
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_tensor_repr(self):
        """
        <b>Description:</b>
        Check TensorEntity class object __repr__ method

        <b>Input data:</b>
        TensorEntity class object with specified "name" and "numpy" parameters

        <b>Expected results:</b>
        Test passes if value returned by __repr__ method is equal to expected
        """
        tensor = self.tensor()
        assert repr(tensor) == "TensorEntity(name=Test Tensor)"
