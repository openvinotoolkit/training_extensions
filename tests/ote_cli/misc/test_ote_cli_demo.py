"""Tests for input parameters with OTE CLI demo tool"""

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

from ote_cli_test_common import (
    create_venv,
    get_some_vars,
    wrong_paths,
    ote_common,
    logger,
    default_train_args_paths
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


class TestDemoCommon:
    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values)
    def create_venv_fx(self, template):
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)

    @pytest.fixture()
    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def get_pretrained_artifacts_fx(self, template, create_venv_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        pretrained_artifact_path = f"{template_work_dir}/trained_{template.model_template_id}"
        command_args = [
            template.model_template_id,
            "--train-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-ann-file"])}',
            "--train-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--train-data-roots"])}',
            "--val-ann-file",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-ann-file"])}',
            "--val-data-roots",
            f'{os.path.join(ote_dir, default_train_args_paths["--val-data-roots"])}',
            "--save-model-to",
            pretrained_artifact_path,
        ]
        command_args.extend(default_train_args_paths["train_params"])
        ote_common(template, root, 'train', command_args)
        assert os.path.exists(pretrained_artifact_path), f"test artifact must exits by path {pretrained_artifact_path}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_no_template(self, back_end, template, create_venv_fx):
        error_string = "ote demo: error: the following arguments are required: template"
        command_args = []
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_no_weights(self, back_end, template, create_venv_fx):
        error_string = "ote demo: error: the following arguments are required: --load-weights"
        command_args = [template.model_template_id,
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_no_input(self, back_end, template, create_venv_fx):
        error_string = "ote demo: error: the following arguments are required: -i/--input"
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'./trained_{template.model_template_id}/weights.pth']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_wrong_weights(self, back_end, template, create_venv_fx):
        error_string = "Path is not valid"
        for case in wrong_paths.values():
            command_args = [template.model_template_id,
                            '--load-weights',
                            case,
                            '--input',
                            f'{os.path.join(ote_dir, "data/airport/train")}']
            ret = ote_common(template, root, 'demo', command_args)
            assert ret['exit_code'] != 0, "Exit code must not be equal 0"
            assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_wrong_input(self, back_end, template, create_venv_fx):
        error_string = "ote demo: error: argument -i/--input: expected one argument"
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'./trained_{template.model_template_id}/weights.pth',
                        '--input']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_fit_size_no_input(self, back_end, template, create_venv_fx):
        error_string = "ote demo: error: argument --fit-to-size: expected 2 arguments"
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'./trained_{template.model_template_id}/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--fit-to-size']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_fit_size_float_input(self, back_end, template, create_venv_fx):
        error_string = "--fit-to-size: invalid int value:"
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'./trained_{template.model_template_id}/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--fit-to-size', '0.0', '0.0']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_fit_size_negative_input(self, back_end, template, create_venv_fx):
        error_string = "Both values of --fit_to_size parameter must be > 0"
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'./trained_{template.model_template_id}/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--fit-to-size', '1', '-1']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_fit_size(self, back_end, template, create_venv_fx, get_pretrained_artifacts_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                        '--input',
                        '--delay',
                        '-1',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--fit-to-size', '1', '1']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_delay_wrong_type(self, back_end, template, create_venv_fx):
        error_string = "invalid int value"
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'./trained_{template.model_template_id}/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--delay',
                        'String']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] != 0, "Exit code must not be equal 0"
        assert error_string in ret['stderr'], f"Different error message {ret['stderr']}"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_loop(self, back_end, template, create_venv_fx, get_pretrained_artifacts_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--delay',
                        '-1',
                        '--loop']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] == 0, "Exit code must not equal 0"

    @e2e_pytest_component
    @pytest.mark.parametrize("back_end, template", params_values, ids=params_ids)
    def test_ote_demo_display_perf(self, back_end, template, create_venv_fx, get_pretrained_artifacts_fx):
        _, template_work_dir, _ = get_some_vars(template, root)
        command_args = [template.model_template_id,
                        '--load-weights',
                        f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                        '--input',
                        f'{os.path.join(ote_dir, "data/airport/train")}',
                        '--delay',
                        '-1',
                        '--display-perf']
        ret = ote_common(template, root, 'demo', command_args)
        assert ret['exit_code'] == 0, "Exit code must not equal 0"
