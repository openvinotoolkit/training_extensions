# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass

import pytest
from otx.v2.adapters.openvino.ops.builder import OperationRegistry
from otx.v2.adapters.openvino.ops.op import Attribute, Operation


class TestOperationRegistry:

    def test(self) -> None:
        registry = OperationRegistry("dummy", add_name_as_attr=True)
        OperationRegistry.REGISTERED_NAME_ATTR

        @dataclass
        class DummyAttributeV1(Attribute):
            pass

        class DummyV1(Operation[DummyAttributeV1]):
            TYPE = "dummy"
            VERSION = "opset1"
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
            VERSION = "opset2"
            ATTRIBUTE_FACTORY = DummyAttributeV2

        registry.register()(DummyV2)
        assert getattr(DummyV2, OperationRegistry.REGISTERED_NAME_ATTR) == "DummyV2"

        assert DummyV1 == registry.get_by_name("DummyV1")
        assert DummyV1 == registry.get_by_type_version("dummy", "opset1")
        assert DummyV2 == registry.get_by_type_version("dummy", "opset2")

        with pytest.raises(KeyError):
            registry.get_by_type_version("dummy", "opset3")
        with pytest.raises(KeyError):
            registry.get_by_type_version("invalid", "opset1")
