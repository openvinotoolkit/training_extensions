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

from ote_sdk.test_suite.pytest_insertions import *  # noqa #pylint: disable=unused-import

pytest_plugins = get_pytest_plugins_from_ote()

ote_conftest_insertion(default_repository_name="ote/training_extensions/")


def pytest_addoption(parser):
    ote_pytest_addoption_insertion(parser)


def pytest_addoption(parser):
    parser.addoption("--algo_be",
                     action="store",
                     type=str,
                     help="--algo_be [ANOMALY_CLASSIFICATION | CLASSIFICATION | DETECTION | SEGMENTATION]")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.algo_be
    if 'algo_be' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("algo_be", [option_value])