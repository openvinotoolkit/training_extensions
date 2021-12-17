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

import json
import os
from subprocess import run

import pytest

from ote_cli.registry import Registry

from tests.ote_cli.common import collect_env_vars, get_some_vars, create_venv, patch_demo_py


args = {
    '--train-ann-file': '',
    '--train-data-roots': 'data/classification/train',
    '--val-ann-file': '',
    '--val-data-roots': 'data/classification/val',
    '--test-ann-files': '',
    '--test-data-roots': 'data/classification/val',
}

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

templates = Registry('external').filter(task_type='CLASSIFICATION').templates
templates_ids = [template.model_template_id for template in templates]

templates = [templates[0]]
templates_ids = [templates_ids[0]]

@pytest.mark.parametrize("template", templates, ids=templates_ids)
def test_ote_train(template):
    work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
    create_venv(algo_backend_dir, work_dir, template_work_dir)
    command_line = ['ote',
                    'train',
                    template.model_template_id,
                    '--train-ann-file',
                    f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                    '--train-data-roots',
                    f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                    '--val-ann-file',
                    f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                    '--val-data-roots',
                    f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                    '--save-model-to',
                    f'{template_work_dir}/trained_{template.model_template_id}',
                    'params',
                    '--learning_parameters.max_num_epochs',
                    '2',
                    '--learning_parameters.batch_size',
                    '2']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/trained_{template.model_template_id}/weights.pth')
    assert os.path.exists(f'{template_work_dir}/trained_{template.model_template_id}/label_schema.json')


@pytest.mark.parametrize("template", templates, ids=templates_ids)
def test_ote_export(template):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'export',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    f'--save-model-to',
                    f'{template_work_dir}/exported_{template.model_template_id}']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml')
    assert os.path.exists(f'{template_work_dir}/exported_{template.model_template_id}/openvino.bin')
    assert os.path.exists(f'{template_work_dir}/exported_{template.model_template_id}/label_schema.json')


@pytest.mark.parametrize("template", templates, ids=templates_ids)
def test_ote_eval(template):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'eval',
                    template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    '--save-performance',
                    f'{template_work_dir}/trained_{template.model_template_id}/performance.json']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/trained_{template.model_template_id}/performance.json')
    

@pytest.mark.parametrize("template", templates, ids=templates_ids)
def test_ote_eval_openvino(template):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'eval',
                    template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    '--save-performance',
                    f'{template_work_dir}/exported_{template.model_template_id}/performance.json']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert os.path.exists(f'{template_work_dir}/exported_{template.model_template_id}/performance.json')
    with open(f'{template_work_dir}/trained_{template.model_template_id}/performance.json') as read_file:
        trained_performance = json.load(read_file)
    with open(f'{template_work_dir}/exported_{template.model_template_id}/performance.json') as read_file:
        exported_performance = json.load(read_file)
        
    for k in trained_performance.keys():
        assert abs(trained_performance[k] - exported_performance[k]) / trained_performance[k] <= 0.01, f"{trained_performance[k]=}, {exported_performance[k]=}"
        

@pytest.mark.parametrize("template", templates, ids=templates_ids)
def test_ote_demo(template):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'demo',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    '--input',
                    f'{os.path.join(ote_dir, args["--test-data-roots"], "0")}',
                    '--delay',
                    '-1']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    

@pytest.mark.parametrize("template", templates, ids=templates_ids)
def test_ote_demo_openvino(template):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = ['ote',
                    'demo',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    '--input',
                    f'{os.path.join(ote_dir, args["--test-data-roots"], "0")}',
                    '--delay',
                    '-1']
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0


@pytest.mark.parametrize("template", templates, ids=templates_ids)
def test_ote_deploy_openvino(template):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    deployment_dir = f'{template_work_dir}/deployed_{template.model_template_id}'
    command_line = ['ote',
                    'deploy',
                    template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    f'--save-model-to',
                    deployment_dir]
    assert run(command_line, env=collect_env_vars(work_dir)).returncode == 0
    assert run(['unzip', 'openvino.zip'],
               cwd=deployment_dir).returncode == 0
    assert run(['python3', '-m', 'venv', 'venv'],
               cwd=os.path.join(deployment_dir, 'python')).returncode == 0
    assert run(['python3', '-m', 'pip', 'install', 'wheel'],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0
    assert run(['python3', '-m', 'pip', 'install', 'demo_package-0.0-py3-none-any.whl'],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0
    patch_demo_py(os.path.join(deployment_dir, 'python', 'demo.py'),
                  os.path.join(deployment_dir, 'python', 'demo_patched.py'))

    assert run(['python3', 'demo_patched.py', '-m', '../model/model.xml', '-i', f'{os.path.join(ote_dir, args["--test-data-roots"], "0")}'],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0
