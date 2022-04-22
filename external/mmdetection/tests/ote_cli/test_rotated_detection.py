"""Tests for rotated object detection with OTE CLI"""

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
    ote_demo_deployment_testing,
    ote_demo_testing,
    ote_demo_openvino_testing,
    ote_deploy_openvino_testing,
    ote_eval_deployment_testing,
    ote_eval_openvino_testing,
    ote_eval_testing,
    ote_train_testing,
    ote_export_testing,
)


args = {
    '--train-ann-file': 'data/car_tree_bug/annotations/instances_default.json',
    '--train-data-roots': 'data/car_tree_bug/images',
    '--val-ann-file': 'data/car_tree_bug/annotations/instances_default.json',
    '--val-data-roots': 'data/car_tree_bug/images',
    '--test-ann-files': 'data/car_tree_bug/annotations/instances_default.json',
    '--test-data-roots': 'data/car_tree_bug/images',
    '--input': 'data/car_tree_bug/images',
    'train_params': [
        'params',
        '--learning_parameters.num_iters',
        '5',
        '--learning_parameters.batch_size',
        '2'
    ]
}

root = '/tmp/ote_cli/'
ote_dir = os.getcwd()

templates = Registry('external/mmdetection').filter(task_type='ROTATED_DETECTION').templates
templates_ids = [template.model_template_id for template in templates]


class TestToolsRotatedDetection:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_train(self, template):
        ote_train_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_export(self, template):
        ote_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval(self, template):
        ote_eval_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_openvino(self, template):
        ote_eval_openvino_testing(template, root, ote_dir, args, threshold=0.1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo(self, template):
        ote_demo_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_openvino(self, template):
        ote_demo_openvino_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_deploy_openvino(self, template):
        ote_deploy_openvino_testing(template, root, ote_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_eval_deployment(self, template):
        ote_eval_deployment_testing(template, root, ote_dir, args, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ote_demo_deployment(self, template):
        ote_demo_deployment_testing(template, root, ote_dir, args)
