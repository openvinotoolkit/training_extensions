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
from subprocess import run
from copy import deepcopy

import pytest
from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component

from ote_cli.registry import Registry

from common import (
    collect_env_vars,
    create_venv,
    get_some_vars,
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_hpo_testing,
    ote_train_testing,
    ote_export_testing,
    pot_optimize_testing,
    pot_eval_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
    wrong_paths
)

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()


class OTECliParamsDemo:
    @pytest.fixture()
    def templates(self, algo_be):
        return Registry('external').filter(task_type=algo_be).templates

    @e2e_pytest_component
    def test_create_venv(self, templates):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    def test_ote_demo_no_weights(self, templates):
        expected_error = "ote demo: error: the following arguments are required: --load-weights"
        for template in templates:
            command_line = ['ote',
                            'demo',
                            template.model_template_id,
                            '--input',
                            f'{os.path.join(ote_dir, "data/airport/train")}']
            assert expected_error in str(run(command_line, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_demo_no_input(self, templates):
        expected_error = "ote demo: error: the following arguments are required: -i/--input"
        for template in templates:
            command_line = ['ote',
                            'demo',
                            template.model_template_id,
                            '--load-weights',
                            './trained_default_template/weights.pth']
            assert expected_error in str(run(command_line, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_demo_wrong_weights(self, templates):
        for template in templates:
            command_line = ['ote',
                            'demo',
                            template.model_template_id,
                            '--load-weights',
                            './trained_default_template/weights.pth',
                            '--input',
                            f'{os.path.join(ote_dir, "data/airport/train")}']
            for case in wrong_paths.values():
                temp = deepcopy(command_line)
                temp[4] = case
                assert "Path is not valid" in str(run(temp, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_demo_wrong_input(self, templates):
        expected_error_line = "ote demo: error: argument -i/--input: expected one argument"
        for template in templates:
            command_line = ['ote',
                            'demo',
                            template.model_template_id,
                            '--load-weights',
                            './trained_default_template/weights.pth',
                            '--input']
            assert expected_error_line in str(run(command_line, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_demo_fit_size_wrong_input(self, templates):
        for template in templates:
            command_line = ['ote',
                            'demo',
                            template.model_template_id,
                            '--load-weights',
                            './trained_default_template/weights.pth',
                            '--input',
                            f'{os.path.join(ote_dir, "data/airport/train")}',
                            '--fit-to-size']
            expected_error_line = "ote demo: error: argument --fit-to-size: expected 2 arguments"
            assert expected_error_line in str(run(command_line, capture_output=True).stderr)
            temp = deepcopy(command_line)
            temp += ['0.0', '0.0']
            expected_error_line = "ote demo: error: argument --fit-to-size: invalid int value: '0.0'"
            assert expected_error_line in str(run(temp, capture_output=True).stderr)
            temp[-1], temp[-2] = "-1", "1"
            expected_error_line = "Both values of --fit_to_size parameter must be > 0"
            assert expected_error_line in str(run(temp, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_demo_delay_wrong_input(self, templates):

        for template in templates:
            command_line = ['ote',
                            'demo',
                            template.model_template_id,
                            '--load-weights',
                            './trained_default_template/weights.pth',
                            '--input',
                            f'{os.path.join(ote_dir, "data/airport/train")}',
                            '--delay',
                            '0.0']
            expected_error_line = "ote demo: error: argument --delay: invalid int value: '0.0'"
            assert expected_error_line in str(run(command_line, capture_output=True).stderr)
            temp = deepcopy(command_line)
            temp[-1] = "-1"
            expected_error_line = "Value of --delay parameter must not be negative"
            assert expected_error_line in str(run(temp, capture_output=True).stderr)

    @e2e_pytest_component
    def test_ote_demo_no_template(self):
        error_string = "ote demo: error: the following arguments are required: template"
        command_line = ['ote',
                        'demo']
        assert error_string in str(run(command_line, capture_output=True).stderr)
