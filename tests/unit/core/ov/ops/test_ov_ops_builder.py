# Copyright (C) 2021-2023 Intel Corporation
# SPDX-Lggggcense-Identifier: Apache-2.0
#

from dataclasses import dataclass

import pytest

from otx.core.ov.ops.builder import OperationRegistry
from otx.core.ov.ops.op import Attribute, Operation
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestOperationRegistry:
    @e2e_pytest_unit
    def test(self):
        registry = OperationRegistry("dummy", add_name_as_attr=True)
        OperationRegistry.REGISTERED_NAME_ATTR

        @dataclass
        class DummyAttributeV1(Attribute):
            pass

        class DummyV1(Operation[DummyAttributeV1]):
            TYPE = "dummy"
            VERSION = 1
            ATTRIBUTE_FACTORY = DummyAttributeV1

        registry.register()(DummyV1)
        assert getattr(DummyV1, OperationRegistry.REGISTERED_NAME_ATTR) == "DummyV1"

        with pytest.raises(KeyError):
            registry.register("another_dummy")(DummyV1)

        @dataclass
        class DummyAttributeV2(Attribute):
            pass

        class DummyV2(Operation[DummyAttributeV2]):
            TYPE = "dummy"
            VERSION = 2
            ATTRIBUTE_FACTORY = DummyAttributeV2

        registry.register()(DummyV2)
        assert getattr(DummyV2, OperationRegistry.REGISTERED_NAME_ATTR) == "DummyV2"

        assert DummyV1 == registry.get_by_name("DummyV1")
        assert DummyV1 == registry.get_by_type_version("dummy", 1)
        assert DummyV2 == registry.get_by_type_version("dummy", 2)

        with pytest.raises(KeyError):
            registry.get_by_type_version("dummy", 3)
        with pytest.raises(KeyError):
            registry.get_by_type_version("invalid", 1)
