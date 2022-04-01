"""Tests for input parameters with OTE CLI find tool"""

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
import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_cli.registry import Registry

from ote_cli.utils.tests import (
    create_venv,
    get_some_vars,
)

from ote_cli_test_common import (
    wrong_paths,
    ote_common,
    logger
)


root = '/tmp/ote_cli/'
ote_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
external_path = os.path.join(ote_dir, "external")


params_values = []
params_ids = []
for back_end_ in ('DETECTION',
                  'CLASSIFICATION',
                  'ANOMALY_CLASSIFICATION',
                  'SEGMENTATION',
                  'ROTATED_DETECTION',
                  'INSTANCE_SEGMENTATION'):
    cur_templates = Registry(external_path).filter(task_type=back_end_).templates
    cur_templates_ids = [template.model_template_id for template in cur_templates]
    params_values += [(back_end_, t) for t in cur_templates]
    params_ids += [back_end_ + ',' + cur_id for cur_id in cur_templates_ids]


class TestFindCommon:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values)
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_cli_find(self, back_end, template, create_venv_fx):
        command_args = []
        ret = ote_common(template, root, "find", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_cli_find_root_same_folder(self, back_end, template, create_venv_fx):
        command_args = ["--root", "."]
        ret = ote_common(template, root, "find", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_cli_find_root_external_folder(self, back_end, template, create_venv_fx):
        command_args = ["--root", external_path]
        ret = ote_common(template, root, "find", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_cli_find_root(self, back_end, template, create_venv_fx):
        command_args = ["--root", template.model_template_path]
        ret = ote_common(template, root, "find", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_cli_task_type(self, back_end, template, create_venv_fx):
        command_args = ["--task_type", back_end]
        ret = ote_common(template, root, "find", command_args)
        assert ret["exit_code"] == 0, "Exit code must be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_cli_find_root_wrong_path(self, back_end, template, create_venv_fx):
        for path in wrong_paths.values():
            command_args = ["--root", path]
            ret = ote_common(template, root, "find", command_args)
            assert ret["exit_code"] != 0, "Exit code must not be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_cli_find_task_type_not_set(self, back_end, template, create_venv_fx):
        command_args = ["--task_id"]
        ret = ote_common(template, root, "find", command_args)
        assert ret["exit_code"] != 0, "Exit code must not be equal 0"
