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

import os

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_cli.utils.tests import check_run

class TestCodeChecks:
    @e2e_pytest_component
    def test_code_checks(self):
        wd = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        check_run(["./tests/run_code_checks.sh"], cwd=wd)
