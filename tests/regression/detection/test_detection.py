"""Tests for Detection with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import pytest
import json
from pathlib import Path

from otx.cli.registry import Registry
from tests.test_suite.run_test_command import (
    nncf_eval_testing,
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_compare,
    otx_eval_openvino_testing,
    otx_export_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing
)
from tests.regression.regression_test_helpers import (
    load_regression_configuration,
    get_result_dict
)

from tests.test_suite.e2e_test_system import e2e_pytest_component

# Configurations for regression test.
REGRESSION_TEST_EPOCHS = "10"
TASK_TYPE = "detection"

otx_dir = os.getcwd()
templates = Registry("otx/algorithms/detection").filter(task_type=TASK_TYPE.upper()).templates
templates_ids = [template.model_template_id for template in templates]

result_dict = get_result_dict(TASK_TYPE)
result_dir = f"/tmp/regression_test_results_{TASK_TYPE}"
Path(result_dir).mkdir(parents=True, exist_ok=True)

# Detection
detection_regression_config = load_regression_configuration(otx_dir, TASK_TYPE, "supervised", "multi_class")
detection_data_args = detection_regression_config["data_path"]
detection_data_args["train_params"] = [
    "params",
    "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS
]

class TestRegressionDetection:
    def setup(self):
        self.label_type = "multi_class"

    def teardown(self):        
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / TASK_TYPE
        otx_train_testing(template, tmp_dir_path, otx_dir, detection_data_args)
        otx_eval_compare(
            template, tmp_dir_path, otx_dir, detection_data_args, 
            detection_regression_config["regression_criteria"], 
            result_dict["train"][TASK_TYPE][self.label_type]["supervised"]
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / f"{TASK_TYPE}/test_semisl"
        config_semisl = load_regression_configuration(otx_dir, TASK_TYPE, "semi_supervised", "multi_class")
        args_semisl = config_semisl["data_path"]

        args_semisl["train_params"] = [
            "params",
            "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS
        ]
        args_semisl["train_params"].extend(["--algo_backend.train_type", "SEMISUPERVISED"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)

        args_semisl.pop("train_params")
        otx_eval_compare(
            template, tmp_dir_path, otx_dir, args_semisl, 
            config_semisl["regression_criteria"],
            result_dict["train"][TASK_TYPE][self.label_type]["semi_supervised"]
        )
    
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, detection_data_args, 
            threshold=0.0, result_dict=result_dict["export"][TASK_TYPE][self.label_type]["supervised"])

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, detection_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, detection_data_args, 
            threshold=0.0, result_dict=result_dict["deploy"][TASK_TYPE][self.label_type]["supervised"])

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, detection_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, tmp_dir_path, otx_dir, detection_data_args, 
            threshold=0.001, result_dict=result_dict["nncf"][TASK_TYPE][self.label_type]["supervised"])

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, detection_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "detection"
        pot_eval_testing(template, tmp_dir_path, otx_dir, detection_data_args,
            result_dict=result_dict["pot"][TASK_TYPE][self.label_type]["supervised"])
