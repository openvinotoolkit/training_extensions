"""Tests for anomaly detection with OTX CLI."""

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
from otx.cli.utils.tests import (
    create_venv,
    get_some_vars,
    nncf_eval_openvino_testing,
    nncf_eval_testing,
    nncf_export_testing,
    nncf_optimize_testing,
    otx_demo_deployment_testing,
    otx_demo_openvino_testing,
    otx_demo_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)
from otx.api.test_suite.e2e_test_system import e2e_pytest_component

from otx.cli.registry import Registry

args = {
    "--train-ann-file": "data/anomaly/detection/train.json",
    "--train-data-roots": "data/anomaly/shapes",
    "--val-ann-file": "data/anomaly/detection/val.json",
    "--val-data-roots": "data/anomaly/shapes",
    "--test-ann-files": "data/anomaly/detection/test.json",
    "--test-data-roots": "data/anomaly/shapes",
    "--input": "data/anomaly/shapes/test/hexagon",
    "train_params": [],
}

root = "/tmp/otx/cli/"
otx_dir = os.getcwd()

templates = Registry("otx/algorithms").filter(task_type="ANOMALY_DETECTION").templates
templates_ids = [template.model_template_id for template in templates]


class TestToolsAnomalyDetection:
    @e2e_pytest_component
    def test_create_venv(self):
        work_dir, _, algo_backend_dir = get_some_vars(templates[0], root)
        create_venv(algo_backend_dir, work_dir)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template):
        otx_train_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template):
        otx_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template):
        otx_eval_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template):
        otx_eval_openvino_testing(template, root, otx_dir, args, threshold=0.01)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo(self, template):
        otx_demo_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo_openvino(self, template):
        otx_demo_openvino_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template):
        otx_deploy_openvino_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template):
        otx_eval_deployment_testing(template, root, otx_dir, args, threshold=0.01)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo_deployment(self, template):
        otx_demo_deployment_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, root)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.xfail(reason="CVS-83124")
    def test_nncf_eval(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        # TODO(AlexanderDokuchaev): return threshold=0.0001 after fix loading NNCF model
        nncf_eval_testing(template, root, otx_dir, args, threshold=0.3)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template):
        pot_optimize_testing(template, root, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template):
        pot_eval_testing(template, root, otx_dir, args)
