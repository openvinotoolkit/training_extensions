"""Tests for Tiling Detection with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import pytest
import json
from pathlib import Path
from timeit import default_timer as timer

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
    get_result_dict,
    REGRESSION_TEST_EPOCHS
)

from tests.test_suite.e2e_test_system import e2e_pytest_component

# Configurations for regression test.
TASK_TYPE = "detection"
TRAIN_TYPE = "tiling"
LABEL_TYPE = "multi_class"

otx_dir = os.getcwd()
templates = Registry("otx/algorithms/detection").filter(task_type=TASK_TYPE.upper()).templates
templates_ids = [template.model_template_id for template in templates]

result_dict = get_result_dict(TASK_TYPE)
result_dir = f"/tmp/regression_test_results/tiling_{TASK_TYPE}"
Path(result_dir).mkdir(parents=True, exist_ok=True)

# Detection
tiling_detection_regression_config = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, LABEL_TYPE)
tiling_detection_data_args = tiling_detection_regression_config["data_path"]
tiling_detection_data_args["train_params"] = [
    "params",
    "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS,
    "--tiling_parameters.enable_tiling", "1",
    "--tiling_parameters.enable_adaptive_params", "1"
]

class TestRegressionTilingDetection:
    def setup_method(self):
        self.label_type = LABEL_TYPE
        self.acc_metric = "Top-1 acc."
        self.train_time = "Train + val time (sec.)"
        self.infer_time = "Infer time (sec.)"
        
        self.export_time = "Export time (sec.)"
        self.export_eval_time = "Export eval time (sec.)"
        
        self.deploy_time = "Deploy time (sec.)"
        self.deploy_eval_time = "Deploy eval time (sec.)"
        
        self.nncf_time = "NNCF time (sec.)"
        self.nncf_eval_time = "NNCF eval time (sec.)"
        
        self.pot_time = "POT time (sec.)"
        self.pot_eval_time = "POT eval time (sec.)"
        
        self.performance = {}
        
    def teardown_method(self):        
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        self.performance[template.name] = {}
        
        tmp_dir_path = tmp_dir_path / TASK_TYPE
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, tiling_detection_data_args)
        train_elapsed_time = timer() - train_start_time
        
        infer_start_time = timer()
        otx_eval_compare(
            template, tmp_dir_path, otx_dir, tiling_detection_data_args, 
            tiling_detection_regression_config["regression_criteria"]["train"], 
            self.performance[template.name],
            self.acc_metric
        )
        infer_elapsed_time = timer() - infer_start_time
    
        self.performance[template.name][self.train_time] = round(train_elapsed_time, 3)
        self.performance[template.name][self.infer_time] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][LABEL_TYPE][TRAIN_TYPE]["train"].append(self.performance)
        
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_eval_openvino(self, template, tmp_dir_path):
        self.performance[template.name] = {}
         
        tmp_dir_path = tmp_dir_path / TASK_TYPE
        export_start_time = timer()
        otx_export_testing(template, tmp_dir_path)
        export_elapsed_time = timer() - export_start_time
        
        export_eval_start_time = timer()
        otx_eval_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            tiling_detection_data_args,
            threshold=1.0,
            criteria=tiling_detection_regression_config["regression_criteria"]["export"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
            acc_metric=self.acc_metric
        )
        export_eval_elapsed_time = timer() - export_eval_start_time
        
        self.performance[template.name][self.export_time] = round(export_elapsed_time, 3)
        self.performance[template.name][self.export_eval_time] = round(export_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["export"].append(self.performance)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_eval_deployment(self, template, tmp_dir_path):
        self.performance[template.name] = {}
        
        tmp_dir_path = tmp_dir_path / TASK_TYPE
        deploy_start_time = timer()
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, tiling_detection_data_args)
        deploy_elapsed_time = timer() - deploy_start_time
        
        deploy_eval_start_time = timer()  
        otx_eval_deployment_testing(
            template,
            tmp_dir_path,
            otx_dir,
            tiling_detection_data_args,
            threshold=1.0,
            criteria=tiling_detection_regression_config["regression_criteria"]["deploy"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
            acc_metric=self.acc_metric
        )
        deploy_eval_elapsed_time = timer() - deploy_eval_start_time
        
        self.performance[template.name][self.deploy_time] = round(deploy_elapsed_time, 3)
        self.performance[template.name][self.deploy_eval_time] = round(deploy_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["deploy"].append(self.performance)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}
        
        tmp_dir_path = tmp_dir_path / TASK_TYPE
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_start_time = timer()
        nncf_optimize_testing(template, tmp_dir_path, otx_dir, tiling_detection_data_args)
        nncf_elapsed_time = timer() - nncf_start_time
        
        nncf_eval_start_time = timer()
        nncf_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            tiling_detection_data_args,
            threshold=1.0,
            criteria=tiling_detection_regression_config["regression_criteria"]["nncf"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
            acc_metric=self.acc_metric
        )
        nncf_eval_elapsed_time = timer() - nncf_eval_start_time
        
        self.performance[template.name][self.nncf_time] = round(nncf_elapsed_time, 3)
        self.performance[template.name][self.nncf_eval_time] = round(nncf_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["nncf"].append(self.performance)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}
        
        tmp_dir_path = tmp_dir_path / TASK_TYPE
        pot_start_time = timer()
        pot_optimize_testing(template, tmp_dir_path, otx_dir, tiling_detection_data_args)
        pot_elapsed_time = timer() - pot_start_time
        
        pot_eval_start_time = timer()
        pot_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            tiling_detection_data_args,
            criteria=tiling_detection_regression_config["regression_criteria"]["nncf"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
            acc_metric=self.acc_metric
        )
        pot_eval_elapsed_time = timer() - pot_eval_start_time
        
        self.performance[template.name][self.nncf_time] = round(pot_elapsed_time, 3)
        self.performance[template.name][self.nncf_eval_time] = round(pot_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["pot"].append(self.performance)