# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

try:
    import e2e.fixtures
    from e2e import config  # noqa
    from e2e.conftest_utils import *  # noqa
    from e2e.conftest_utils import pytest_addoption as _e2e_pytest_addoption  # noqa
    from e2e.utils import get_plugins_from_packages

    pytest_plugins = get_plugins_from_packages([e2e])
except ImportError:
    _e2e_pytest_addoption = None
    pass
import pytest

from tests.test_suite.pytest_insertions import (
    get_pytest_plugins_from_otx,
    otx_conftest_insertion,
    otx_pytest_addoption_insertion,
    otx_pytest_generate_tests_insertion,
)
from tests.test_suite.training_tests_common import REALLIFE_USECASE_CONSTANT

pytest_plugins = get_pytest_plugins_from_otx()

otx_conftest_insertion(default_repository_name="otx/training_extensions/external/anomaly")


@pytest.fixture
def otx_test_domain_fx():
    raise NotImplementedError("Please, implement the fixture otx_test_domain_fx in your test file")


@pytest.fixture
def otx_test_scenario_fx(current_test_parameters_fx):
    assert isinstance(current_test_parameters_fx, dict)
    if current_test_parameters_fx.get("usecase") == REALLIFE_USECASE_CONSTANT:
        return "performance"
    else:
        return "integration"


@pytest.fixture(scope="session")
def otx_templates_root_dir_fx():
    import logging
    import os.path as osp

    logger = logging.getLogger(__name__)
    root = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root = osp.realpath(f"{root}/../../otx/algorithms/anomaly/configs")
    logger.debug(f"overloaded otx_templates_root_dir_fx: return {root}")
    return root


@pytest.fixture(scope="session")
def otx_reference_root_dir_fx():
    import logging
    import os.path as osp

    logger = logging.getLogger(__name__)
    root = osp.dirname(osp.realpath(__file__))
    root = f"{root}/reference/"
    logger.debug(f"overloaded otx_reference_root_dir_fx: return {root}")
    return root


# pytest magic
def pytest_generate_tests(metafunc):
    otx_pytest_generate_tests_insertion(metafunc)


def pytest_addoption(parser):
    otx_pytest_addoption_insertion(parser)
