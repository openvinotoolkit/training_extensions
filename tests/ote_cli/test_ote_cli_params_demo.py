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

from subprocess import run
from copy import deepcopy

import os
import pytest

from ote_sdk.test_suite.e2e_test_system import e2e_pytest_component
from ote_cli.registry import Registry

from common import (
    create_venv,
    get_some_vars,
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_train_testing,
    ote_export_testing,
    pot_optimize_testing,
    pot_eval_testing,
    nncf_optimize_testing,
    nncf_export_testing,
    nncf_eval_testing,
    nncf_eval_openvino_testing,
    wrong_paths,
    ote_demo_common
)


root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

templates = Registry('external').templates
templates_ids = [template.model_template_id for template in templates]


class TestOTECliDemoParams:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_no_template(self, template):
        error_string = "ote demo: error: the following arguments are required: template"
        ret = ote_demo_common(template, root, [])
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_no_weights(self, template):
        error_string = "ote demo: error: the following arguments are required: --load-weights"
        command_args = [template.model_template_id,
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}']
        ret = ote_demo_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_no_input(self, template):
        error_string = "ote demo: error: the following arguments are required: -i/--input"
        command_args = [template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth']
        ret = ote_demo_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_wrong_weights(self, template):
        command_args = [template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}']
        for case in wrong_paths.values():
            temp = deepcopy(command_args)
            temp[2] = case
            ret = ote_demo_common(template, root, temp)
            assert "Path is not valid" in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_wrong_input(self, template):
        error_string = "ote demo: error: argument -i/--input: expected one argument"
        command_args = [template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input']
        ret = ote_demo_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_fit_size_no_input(self, template):
        error_string = "ote demo: error: argument --fit-to-size: expected 2 arguments"
        command_args = [template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--fit-to-size']
        ret = ote_demo_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_fit_size_float_input(self, template):
        error_string = "--fit-to-size: invalid int value:"
        command_args = [template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--fit-to-size', '0.0', '0.0']
        ret = ote_demo_common(template, root, command_args)
        assert error_string in str(ret.stderr)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_fit_size_negative_input(self, template):
        error_string = "Both values of --fit_to_size parameter must be > 0"
        command_args = [template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--fit-to-size', '1', '-1']
        ret = ote_demo_common(template, root, command_args)
        assert error_string in str(ret.stderr)
