# Copyright (C) 2021-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from otx.core.patcher import Patcher
from tests.test_suite.e2e_test_system import e2e_pytest_unit


class MockClass:
    def method(self, *args, **kwargs):
        pass

    @staticmethod
    def static_method(*args, **kwargs):
        pass

    @classmethod
    def class_method(cls, *args, **kwargs):
        pass


def mock_function(*args, **kwargs):
    pass


class Counter:
    def __init__(self):
        self._ctr = 0

    def inc(self):
        self._ctr += 1

    def __eq__(self, other):
        return self._ctr == other


class TestPatcher:
    @e2e_pytest_unit
    def test_patch(self):
        def dummy_wrapper(ctx, fn, *args, **kwargs):
            kwargs.get("ctr").inc()
            return fn(*args, **kwargs)

        def test_instance():
            patcher = Patcher()
            ctr = Counter()
            mock_class = MockClass()

            patcher.patch(mock_class.method, dummy_wrapper)
            patcher.patch(mock_class.method, dummy_wrapper)
            mock_class.method(ctr=ctr)
            assert len(patcher._patched) == 1
            assert len(list(patcher._patched.values())[-1]) == 1
            assert ctr == 1
            patcher.unpatch(mock_class.method)
            assert len(patcher._patched) == 0

            patcher.patch(mock_class.method, dummy_wrapper)
            patcher.patch(mock_class.method, dummy_wrapper, force=False)
            mock_class.method(ctr=ctr)
            assert len(patcher._patched) == 1
            assert len(list(patcher._patched.values())[-1]) == 2
            assert ctr == 3
            patcher.unpatch(mock_class.method, 1)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(mock_class.method, dummy_wrapper, force=False)
            patcher.patch(mock_class.method, dummy_wrapper, force=False)
            assert len(list(patcher._patched.values())[-1]) == 3
            patcher.unpatch(mock_class.method, 2)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(mock_class.static_method, dummy_wrapper)
            mock_class.static_method(ctr=ctr)
            assert len(patcher._patched) == 2
            assert len(list(patcher._patched.values())[-1]) == 1
            assert ctr == 4
            patcher.unpatch(mock_class.static_method)
            assert len(patcher._patched) == 1

            patcher.patch(mock_class.static_method, dummy_wrapper)
            patcher.patch(mock_class.static_method, dummy_wrapper, force=False)
            mock_class.static_method(ctr=ctr)
            assert len(patcher._patched) == 2
            assert len(list(patcher._patched.values())[-1]) == 2
            assert ctr == 6
            patcher.unpatch(mock_class.static_method, 1)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(mock_class.static_method, dummy_wrapper, force=False)
            patcher.patch(mock_class.static_method, dummy_wrapper, force=False)
            assert len(list(patcher._patched.values())[-1]) == 3
            patcher.unpatch(mock_class.static_method, 2)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(mock_class.class_method, dummy_wrapper)
            mock_class.class_method(ctr=ctr)
            assert len(patcher._patched) == 3
            assert len(list(patcher._patched.values())[-1]) == 1
            assert ctr == 7
            patcher.unpatch(mock_class.class_method)
            assert len(patcher._patched) == 2

            patcher.patch(mock_class.class_method, dummy_wrapper)
            patcher.patch(mock_class.class_method, dummy_wrapper, force=False)
            mock_class.class_method(ctr=ctr)
            assert len(patcher._patched) == 3
            assert len(list(patcher._patched.values())[-1]) == 2
            assert ctr == 9
            patcher.unpatch(mock_class.class_method, 1)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(mock_class.class_method, dummy_wrapper, force=False)
            patcher.patch(mock_class.class_method, dummy_wrapper, force=False)
            assert len(list(patcher._patched.values())[-1]) == 3
            patcher.unpatch(mock_class.class_method, 2)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.unpatch()
            assert len(patcher._patched) == 0

        def test_class():
            patcher = Patcher()
            ctr = Counter()
            mock_class = MockClass()

            patcher.patch(MockClass.method, dummy_wrapper)
            patcher.patch(MockClass.method, dummy_wrapper)
            mock_class.method(ctr=ctr)
            assert len(patcher._patched) == 1
            assert len(list(patcher._patched.values())[-1]) == 1
            assert ctr == 1
            patcher.unpatch(MockClass.method)
            assert len(patcher._patched) == 0

            patcher.patch(MockClass.method, dummy_wrapper)
            patcher.patch(MockClass.method, dummy_wrapper, force=False)
            mock_class.method(ctr=ctr)
            assert len(patcher._patched) == 1
            assert len(list(patcher._patched.values())[-1]) == 2
            assert ctr == 3
            patcher.unpatch(MockClass.method, 1)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(MockClass.method, dummy_wrapper, force=False)
            patcher.patch(MockClass.method, dummy_wrapper, force=False)
            assert len(list(patcher._patched.values())[-1]) == 3
            patcher.unpatch(MockClass.method, 2)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(MockClass.static_method, dummy_wrapper)
            mock_class.static_method(ctr=ctr)
            assert len(patcher._patched) == 2
            assert len(list(patcher._patched.values())[-1]) == 1
            assert ctr == 4
            patcher.unpatch(MockClass.static_method)
            assert len(patcher._patched) == 1

            patcher.patch(MockClass.static_method, dummy_wrapper)
            patcher.patch(MockClass.static_method, dummy_wrapper, force=False)
            mock_class.static_method(ctr=ctr)
            assert len(patcher._patched) == 2
            assert len(list(patcher._patched.values())[-1]) == 2
            assert ctr == 6
            patcher.unpatch(MockClass.static_method, 1)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(MockClass.static_method, dummy_wrapper, force=False)
            patcher.patch(MockClass.static_method, dummy_wrapper, force=False)
            assert len(list(patcher._patched.values())[-1]) == 3
            patcher.unpatch(MockClass.static_method, 2)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(MockClass.class_method, dummy_wrapper)
            mock_class.class_method(ctr=ctr)
            assert len(patcher._patched) == 3
            assert len(list(patcher._patched.values())[-1]) == 1
            assert ctr == 7
            patcher.unpatch(MockClass.class_method)
            assert len(patcher._patched) == 2

            patcher.patch(MockClass.class_method, dummy_wrapper)
            patcher.patch(MockClass.class_method, dummy_wrapper, force=False)
            mock_class.class_method(ctr=ctr)
            assert len(patcher._patched) == 3
            assert len(list(patcher._patched.values())[-1]) == 2
            assert ctr == 9
            patcher.unpatch(MockClass.class_method, 1)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.patch(MockClass.class_method, dummy_wrapper, force=False)
            patcher.patch(MockClass.class_method, dummy_wrapper, force=False)
            assert len(list(patcher._patched.values())[-1]) == 3
            patcher.unpatch(MockClass.class_method, 2)
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.unpatch()
            assert len(patcher._patched) == 0

        def test_module():
            patcher = Patcher()
            ctr = Counter()

            patcher.patch(
                "tests.unit.core.test_core_patcher.mock_function",
                dummy_wrapper,
            )
            patcher.patch(
                "tests.unit.core.test_core_patcher.mock_function",
                dummy_wrapper,
            )
            from tests.unit.core.test_core_patcher import mock_function

            mock_function(ctr=ctr)
            assert len(patcher._patched) == 1
            assert len(list(patcher._patched.values())[-1]) == 1
            assert ctr == 1
            patcher.unpatch("tests.unit.core.test_core_patcher.mock_function")
            assert len(patcher._patched) == 0

            patcher.patch(
                "tests.unit.core.test_core_patcher.mock_function",
                dummy_wrapper,
            )
            patcher.patch(
                "tests.unit.core.test_core_patcher.mock_function",
                dummy_wrapper,
                force=False,
            )
            from tests.unit.core.test_core_patcher import mock_function

            mock_function(ctr=ctr)
            assert len(patcher._patched) == 1
            assert len(list(patcher._patched.values())[-1]) == 2
            assert ctr == 3
            patcher.unpatch(
                "tests.unit.core.test_core_patcher.mock_function",
                1,
            )
            assert len(list(patcher._patched.values())[-1]) == 1

            patcher.unpatch()
            assert len(patcher._patched) == 0

        test_instance()
        test_class()
        test_module()

    @e2e_pytest_unit
    def test_import_obj(self):
        patcher = Patcher()
        assert (Patcher, "patch") == patcher.import_obj("otx.core.patcher.Patcher.patch")
        assert (Patcher, "patch") == patcher.import_obj(Patcher.patch)
