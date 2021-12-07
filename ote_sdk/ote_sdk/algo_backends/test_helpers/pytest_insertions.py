# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""
This file contains functions that may be used in the conftest file of algo
backend and in the standard pytest hooks:
* pytest_addoption
* pytest_generate_tests
to add to pytest functionality required for algo backends reallife training tests.
"""

try:
    import e2e.fixtures
    from e2e import config  # noqa
    from e2e.conftest_utils import *  # noqa
    from e2e.conftest_utils import pytest_addoption as _e2e_pytest_addoption  # noqa
    from e2e.utils import get_plugins_from_packages

    _pytest_plugins_from_e2e = get_plugins_from_packages([e2e])
except ImportError:
    _e2e_pytest_addoption = None
    _pytest_plugins_from_e2e = []


def get_pytest_plugins_from_ote():
    """
    The function generates pytest_plugins variable that should be used
    in an algo backend' conftest.py file.
    """
    import ote_sdk.algo_backends.test_helpers.fixtures  # noqa

    pytest_plugins_from_ote_sdk = ["ote_sdk.algo_backends.test_helpers.fixtures"]
    pytest_plugins = list(_pytest_plugins_from_e2e) + pytest_plugins_from_ote_sdk
    return pytest_plugins


def ote_pytest_addoption_insertion(parser):
    """
    The function should be called in the standard pytest hook pytest_addoption
    to add the options required for reallife training tests.
    """
    if _e2e_pytest_addoption:
        _e2e_pytest_addoption(parser)
    parser.addoption(
        "--dataset-definitions",
        action="store",
        default=None,
        help="Path to the dataset_definitions.yml file for tests that require datasets.",
    )
    parser.addoption(
        "--test-usecase",
        action="store",
        default=None,
        help="Optional. If the parameter is set, it filters test_ote_training tests by usecase field.",
    )
    parser.addoption(
        "--expected-metrics-file",
        action="store",
        default=None,
        help="Optional. If the parameter is set, it points the YAML file with expected test metrics.",
    )


def ote_pytest_generate_tests_insertion(metafunc):
    """
    The function should be called in the standard pytest hook pytest_generate_tests
    in algo backend's conftest.py file to generate parameters of reallife training tests.
    """
    import logging

    from .training_tests_helper import OTETrainingTestInterface

    logger = logging.getLogger(__name__)
    if metafunc.cls is None:
        return False
    if not issubclass(metafunc.cls, OTETrainingTestInterface):
        return False

    logger.debug(f"ote_pytest_generate_tests_insertion: begin handling {metafunc.cls}")

    # It allows to filter by usecase
    usecase = metafunc.config.getoption("--test-usecase")

    argnames, argvalues, ids = metafunc.cls.get_list_of_tests(usecase)

    assert isinstance(argnames, (list, tuple))
    assert "test_parameters" in argnames
    assert isinstance(argvalues, list)
    assert isinstance(ids, list)
    assert len(argvalues) == len(ids)
    assert all(isinstance(v, str) for v in ids)

    metafunc.parametrize(argnames, argvalues, ids=ids, scope="class")
    logger.debug(f"ote_pytest_generate_tests_insertion: end handling {metafunc.cls}")
    return True


def ote_conftest_insertion(*, default_repository_name=""):
    """
    The function should be called in an algo backend's conftest.py file
    to set default repository name in e2e- test system.
    """
    try:
        import os

        from e2e import config as config_e2e

        config_e2e.repository_name = os.environ.get(
            "TT_REPOSITORY_NAME", default_repository_name
        )
    except ImportError:
        pass
