"""OTX V2 API-core Unit-Test codes (Registry)."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.v2.api.core.registry import BaseRegistry


class TestBaseRegistry:
    """
    This class contains unit tests for the BaseRegistry class.
    """

    def test_register_module(self) -> None:
        """
        Test case to verify the register_module method of BaseRegistry class.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with name "test_module" and module TestBaseRegistry.
        3. Verify that "test_module" is present in the module_dict of the registry.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(name="test_module", module=TestBaseRegistry)
        assert "test_module" in registry.module_dict

    def test_register_module_with_type(self) -> None:
        """
        Test case to verify the register_module method of BaseRegistry class with type_name parameter.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with type_name "test_type", name "test_module" and module TestBaseRegistry.
        3. Verify that "test_module" is present in the registry_dict["test_type"].module_dict of the registry.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(type_name="test_type", name="test_module", module=TestBaseRegistry)
        assert "test_module" in registry.registry_dict["test_type"].module_dict

    def test_register_module_wo_type_name_and_name(self) -> None:
        """
        Test case to verify the register_module method of BaseRegistry class without type_name and name parameters.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with module TestBaseRegistry.
        3. Verify that "TestBaseRegistry" is present in the module_dict of the registry.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(module=TestBaseRegistry)
        assert "TestBaseRegistry" in registry.module_dict

    def test_register_module_with_existing_name(self) -> None:
        """
        Test case to verify the register_module method of BaseRegistry class with an existing name.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with name "test_module" and module TestBaseRegistry.
        3. Try to register another module with the same name and module TestBaseRegistry.
        4. Verify that a KeyError is raised.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(name="test_module", module=TestBaseRegistry)
        with pytest.raises(KeyError):
            registry.register_module(name="test_module", module=TestBaseRegistry)

    def test_register_module_not_class_or_function(self) -> None:
        """
        Test case to verify the register_module method of BaseRegistry class with a non-class or non-function module.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with name "test_module" and module TestBaseRegistry.
        3. Try to register another module with name "test_module_2" and module TestBaseRegistry().
        4. Verify that a TypeError is raised.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(name="test_module", module=TestBaseRegistry)
        with pytest.raises(TypeError):
            registry.register_module(name="test_module_2", module=TestBaseRegistry())

    def test_register_module_with_existing_name_and_force(self) -> None:
        """
        Test case to verify the register_module method of BaseRegistry class with an existing name and force parameter.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with name "test_module" and module TestBaseRegistry.
        3. Try to register another module with the same name and module TestBaseRegistry with force=True.
        4. Verify that "test_module" is present in the module_dict of the registry.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(name="test_module", module=TestBaseRegistry)
        registry.register_module(name="test_module", module=TestBaseRegistry, force=True)
        assert "test_module" in registry.module_dict

    def test_get_with_module_dict(self) -> None:
        """
        Test case to verify the get method of BaseRegistry class with module_dict parameter.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with name "test_module" and module TestBaseRegistry.
        3. Call the get method with "test_module" as parameter.
        4. Verify that the returned value is TestBaseRegistry.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(name="test_module", module=TestBaseRegistry)
        assert registry.get("test_module") == TestBaseRegistry

    def test_get_with_registry_dict(self) -> None:
        """
        Test case to verify the get method of BaseRegistry class with registry_dict parameter.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with type_name "test_type", name "test_module" and module TestBaseRegistry.
        3. Call the get method with "test_type" as parameter.
        4. Verify that the returned value has name "test_type".
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(type_name="test_type", name="test_module", module=TestBaseRegistry)
        assert registry.get("test_type").name == "test_type"

    def test_get_from_all_registry(self) -> None:
        """
        Test case to verify the get method of BaseRegistry class with multiple registry_dict parameters.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with type_name "sub_1", name "sub_1_module_1" and module TestBaseRegistry.
        3. Register a module with type_name "sub_2", name "sub_2_module_1" and module TestBaseRegistry.
        4. Call the get method with "sub_2_module_1" as parameter.
        5. Verify that the returned value is TestBaseRegistry.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(type_name="sub_1", name="sub_1_module_1", module=TestBaseRegistry)
        registry.register_module(type_name="sub_2", name="sub_2_module_1", module=TestBaseRegistry)
        assert registry.get("sub_2_module_1") == TestBaseRegistry

    def test_get_with_missing_module(self) -> None:
        """
        Test case to verify the get method of BaseRegistry class with a missing module.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Call the get method with "missing_module" as parameter.
        3. Verify that the returned value is None.
        """

        registry = BaseRegistry("test_registry")
        assert registry.get("missing_module") is None

    def test_len(self) -> None:
        """
        Test case to verify the len method of BaseRegistry class.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with name "test_module" and module TestBaseRegistry.
        3. Verify that the length of the registry is 1.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(name="test_module", module=TestBaseRegistry)
        assert len(registry) == 1

    def test_contains_with_existing_module(self) -> None:
        """
        Test case to verify the contains method of BaseRegistry class with an existing module.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with name "test_module" and module TestBaseRegistry.
        3. Verify that "test_module" is present in the registry.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(name="test_module", module=TestBaseRegistry)
        assert "test_module" in registry

    def test_contains_with_missing_module(self) -> None:
        """
        Test case to verify the contains method of BaseRegistry class with a missing module.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Verify that "missing_module" is not present in the registry.
        """

        registry = BaseRegistry("test_registry")
        assert "missing_module" not in registry

    def test_repr(self) -> None:
        """
        Test case to verify the __repr__ method of BaseRegistry class.

        Steps:
        1. Create an instance of BaseRegistry class.
        2. Register a module with name "test_module" and module TestBaseRegistry.
        3. Register a module with type_name "test_type", name "test_module_2" and module TestBaseRegistry.
        4. Verify that "test_registry", "test_module", "test_type" and "test_module_2" are present in the __repr__ output.
        """

        registry = BaseRegistry("test_registry")
        registry.register_module(name="test_module", module=TestBaseRegistry)
        registry.register_module(type_name="test_type", name="test_module_2", module=TestBaseRegistry)
        assert "test_registry" in registry.__repr__()
        assert "test_module" in registry.__repr__()
        assert "test_type" in registry.__repr__()
        assert "test_module_2" in registry.__repr__()
