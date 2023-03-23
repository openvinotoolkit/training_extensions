# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import pytest

from otx.core.ov.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class TestRegistry:
    @e2e_pytest_unit
    def test_register(self):
        registry = Registry("dummy")

        @registry.register()
        def dummy1():
            pass

        assert "dummy1" in registry.registry_dict.keys()
        assert dummy1 in registry
        assert dummy1 == registry.get("dummy1")

        @registry.register("dummy_name")
        def dummy2():
            pass

        assert "dummy_name" in registry.registry_dict.keys()
        assert dummy2 in registry
        assert dummy2 == registry.get("dummy_name")

        with pytest.raises(KeyError):
            registry.get("error")
