"""Tests for input parameters with OTE CLI export tool"""
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

from common import (
    create_venv,
    get_some_vars,
    wrong_paths,
<<<<<<< HEAD
<<<<<<< HEAD
    ote_common,
    logger
=======
    ote_common
>>>>>>> c9e7fcf1 (test ote cli common args)
=======
    ote_common,
    logger
>>>>>>> 070c0d85 (iteraiton 001)
)


root = '/tmp/ote_cli/'
<<<<<<< HEAD
<<<<<<< HEAD
ote_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
=======
ote_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
>>>>>>> c9e7fcf1 (test ote cli common args)
=======
ote_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
>>>>>>> 070c0d85 (iteraiton 001)
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


class TestExportCommon:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values)
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_export_no_template(self, back_end, template, create_venv_fx):
        error_string = "ote export: error: the following arguments are required:" \
                       " template, --load-weights, --save-model-to"
        command_line = []
        ret = ote_common(template, root, 'export', command_line)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_export_no_weights(self, back_end, template, create_venv_fx):
        error_string = "ote export: error: the following arguments are required: --load-weights"
        command_line = [template.model_template_id,
                        f'--save-model-to',
                        f'./exported_{template.model_template_id}']
        ret = ote_common(template, root, 'export', command_line)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_export_no_save_to(self, back_end, template, create_venv_fx):
        error_string = "ote export: error: the following arguments are required: --save-model-to"
        command_line = [template.model_template_id,
                        '--load-weights',
                        f'./trained_{template.model_template_id}/weights.pth']
        ret = ote_common(template, root, 'export', command_line)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_export_wrong_path_load_weights(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--load-weights',
                            case,
                            f'--save-model-to',
                            f'./exported_{template.model_template_id}']
            ret = ote_common(template, root, 'export', command_line)
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_export_wrong_path_save_model_to(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_line = [template.model_template_id,
                            '--load-weights',
                            f'./trained_{template.model_template_id}/weights.pth',
                            f'--save-model-to',
                            case]
            ret = ote_common(template, root, 'export', command_line)
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"
