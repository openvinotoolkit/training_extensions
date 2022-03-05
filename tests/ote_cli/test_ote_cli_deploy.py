"""Tests for input parameters with OTE CLI deploy tool"""
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
    ote_common
)


root = '/tmp/ote_cli/'
ote_dir = os.getcwd()


params_values = []
params_ids = []
for back_end in ('DETECTION', 'CLASSIFICATION', 'ANOMALY_CLASSIFICATION', 'SEGMENTATION'):
    cur_templates = Registry('external').filter(task_type=back_end).templates
    cur_templates_ids = [template.model_template_id for template in cur_templates]
    params_values += [(back_end, t) for t in cur_templates]
    params_ids += [back_end + ',' + cur_id for cur_id in cur_templates_ids]


class TestDeployCommon:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("domain, template", params_values)
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("domain, template", params_values, ids=params_ids)
    def test_ote_deploy_no_template(self, domain, template, create_venv_fx):
        error_string = "ote demo: error: the following arguments are required: template"
        ret = ote_common(template, root, "deploy", [])
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("domain, template", params_values, ids=params_ids)
    def test_ote_deploy_no_weights(self, domain, template, create_venv_fx):
        error_string = "ote deploy: error: the following arguments are required: --load-weights"
        command_args = [template.model_template_id,
                        f'--save-model-to',
                        f'./deployed_{template.model_template_id}']
        ret = ote_common(template, root, "deploy", command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("domain, template", params_values, ids=params_ids)
    def test_ote_deploy_wrong_paths(self, domain, template, create_venv_fx):
        error_string = "Path is not valid"
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'./exported_{template.model_template_id}/openvino.xml',
                        '--save-model-to',
                        f'./deployed_{template.model_template_id}']
        for case in wrong_paths.values():
            temp = deepcopy(command_args)
            temp[4] = case
            ret = ote_common(template, root, "deploy", temp)
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"
