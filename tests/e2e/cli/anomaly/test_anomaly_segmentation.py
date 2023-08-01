"""Tests for anomaly segmentation with OTX CLI"""

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

from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    nncf_eval_openvino_testing,
    nncf_eval_testing,
    nncf_export_testing,
    nncf_optimize_testing,
    nncf_validate_fq_testing,
    otx_demo_deployment_testing,
    otx_demo_openvino_testing,
    otx_demo_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_train_testing,
    ptq_eval_testing,
    ptq_optimize_testing,
    ptq_validate_fq_testing,
)

args = {
    "--train-data-roots": "tests/assets/anomaly/hazelnut/train",
    "--val-data-roots": "tests/assets/anomaly/hazelnut/test",
    "--test-data-roots": "tests/assets/anomaly/hazelnut/test",
    "--input": "tests/assets/anomaly/hazelnut/test/colour",
    "train_params": [],
}

otx_dir = os.getcwd()

templates = Registry("src/otx/algorithms").filter(task_type="ANOMALY_SEGMENTATION").templates
templates_ids = [template.model_template_id for template in templates]


class TestToolsAnomalySegmentation:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args, deterministic=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        otx_export_testing(template, tmp_dir_path, check_ir_meta=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=0.2)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo(self, template, tmp_dir_path):
        otx_demo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo_openvino(self, template, tmp_dir_path):
        otx_demo_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo_deployment(self, template, tmp_dir_path):
        otx_demo_deployment_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_validate_fq(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_validate_fq_testing(template, tmp_dir_path, otx_dir, "anomaly", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, tmp_dir_path, otx_dir, args, threshold=0.01)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_optimize(self, template, tmp_dir_path):
        ptq_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_validate_fq(self, template, tmp_dir_path):
        ptq_validate_fq_testing(template, tmp_dir_path, otx_dir, "anomaly", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_eval(self, template, tmp_dir_path):
        ptq_eval_testing(template, tmp_dir_path, otx_dir, args)
