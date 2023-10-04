# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest
from otx.v2.adapters.openvino.registry import Registry


class TestRegistry:
    def test_register(self) -> None:
        registry = Registry("dummy")

        @registry.register()
        def dummy1() -> None:
            pass

        assert "dummy1" in registry.registry_dict
        assert dummy1 in registry
        assert dummy1 == registry.get("dummy1")

        @registry.register("dummy_name")
        def dummy2() -> None:
            pass

        assert "dummy_name" in registry.registry_dict
        assert dummy2 in registry
        assert dummy2 == registry.get("dummy_name")

        with pytest.raises(KeyError):
            registry.get("error")
