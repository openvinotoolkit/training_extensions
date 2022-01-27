"""Tests for object detection with OTE CLI"""

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
from subprocess import run

from constants.ote_cli_components import OteCliComponent
from constants.requirements import Requirements

from ote_cli.registry import Registry
from tests.ote_cli.common import (
    collect_env_vars,
    create_venv,
    get_some_vars,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_hpo_testing,
    ote_train_testing,
    ote_export_testing,
)


args = {
    '--train-ann-file': 'data/airport/annotation_example_train.json',
    '--train-data-roots': 'data/airport/train',
    '--val-ann-file': 'data/airport/annotation_example_train.json',
    '--val-data-roots': 'data/airport/train',
    '--test-ann-files': 'data/airport/annotation_example_train.json',
    '--test-data-roots': 'data/airport/train',
    '--input': 'data/airport/train',
    'train_params': [
        'params',
        '--learning_parameters.num_iters',
        '2',
        '--learning_parameters.batch_size',
        '2'
    ]
}

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

templates = Registry('external').filter(task_type='DETECTION').templates
templates_ids = [template.model_template_id for template in templates]

@pytest.mark.components(OteCliComponent.OTE_CLI)
class TestToolsDetection:
    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_create_venv():
        work_dir, template_work_dir, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir, template_work_dir)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train(template):
        ote_train_testing(template, root, ote_dir, args)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_export(template):
        ote_export_testing(template, root)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval(template):
        ote_eval_testing(template, root, ote_dir, args)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_openvino(template):
        ote_eval_openvino_testing(template, root, ote_dir, args, threshold=0.01)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo(template):
        ote_demo_testing(template, root, ote_dir, args)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_openvino(template):
        ote_demo_openvino_testing(template, root, ote_dir, args)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_deploy_openvino(template):
        ote_deploy_openvino_testing(template, root, ote_dir, args)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_hpo(template):
        ote_hpo_testing(template, root, ote_dir, args)


    @pytest.mark.priority_medium
    @pytest.mark.component
    @pytest.mark.reqids(Requirements.REQ_1)
    def test_notebook():
        work_dir = os.path.join(root, 'DETECTION')
        assert run(['pytest', '--nbmake', 'ote_cli/notebooks/train.ipynb', '-v'], env=collect_env_vars(work_dir)).returncode == 0
