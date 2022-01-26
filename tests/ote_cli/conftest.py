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


from ote_sdk.test_suite.pytest_insertions import (
    get_pytest_plugins_from_ote,
    ote_conftest_insertion,
    ote_pytest_addoption_insertion,
    ote_pytest_generate_tests_insertion,
)

pytest_plugins = get_pytest_plugins_from_ote()

ote_conftest_insertion(default_repository_name="ote/training_extensions/")


# pytest magic
def pytest_generate_tests(metafunc):
    ote_pytest_generate_tests_insertion(metafunc)


def pytest_addoption(parser):
    ote_pytest_addoption_insertion(parser)
