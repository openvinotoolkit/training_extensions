# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

try:
    import e2e.fixtures

    from e2e.conftest_utils import * # noqa
    from e2e.conftest_utils import pytest_addoption as _e2e_pytest_addoption # noqa
    from e2e import config # noqa
    from e2e.utils import get_plugins_from_packages
    pytest_plugins = get_plugins_from_packages([e2e])
except ImportError:
    _e2e_pytest_addoption = None
    pass
import config
import pytest
from ote_sdk.test_suite.pytest_insertions import *
from ote_sdk.test_suite.training_tests_common import REALLIFE_USECASE_CONSTANT

pytest_plugins = get_pytest_plugins_from_ote()

ote_conftest_insertion(default_repository_name='ote/training_extensions/external/model-preparation-algorithm')

@pytest.fixture
def ote_test_domain_fx():
    return 'model-preparation-algorithm'

@pytest.fixture
def ote_test_scenario_fx(current_test_parameters_fx):
    assert isinstance(current_test_parameters_fx, dict)
    if current_test_parameters_fx.get('usecase') == REALLIFE_USECASE_CONSTANT:
        return 'performance'
    else:
        return 'integration'

@pytest.fixture(scope='session')
def ote_templates_root_dir_fx():
    import os.path as osp
    import logging
    logger = logging.getLogger(__name__)
    root = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root = f'{root}/configs/'
    logger.debug(f'overloaded ote_templates_root_dir_fx: return {root}')
    return root

@pytest.fixture(scope='session')
def ote_reference_root_dir_fx():
    import os.path as osp
    import logging
    logger = logging.getLogger(__name__)
    root = osp.dirname(osp.dirname(osp.realpath(__file__)))
    root = f'{root}/tests/reference/'
    logger.debug(f'overloaded ote_reference_root_dir_fx: return {root}')
    return root

# pytest magic
def pytest_generate_tests(metafunc):
    ote_pytest_generate_tests_insertion(metafunc)

def pytest_addoption(parser):
    ote_pytest_addoption_insertion(parser)
