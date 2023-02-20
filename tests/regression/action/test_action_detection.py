"""Tests for Action Detection with OTX CLI."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os
import pytest
import json
from pathlib import Path
from timeit import default_timer as timer


from otx.cli.registry import Registry
from tests.regression.regression_test_helpers import (
    get_result_dict,
    load_regression_configuration,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_eval_compare,
    otx_eval_openvino_testing,
    otx_export_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)

# Configurations for regression test.
REGRESSION_TEST_EPOCHS = "1"
TASK_TYPE = "action_detection"
TRAIN_TYPE = "supervised"
LABEL_TYPE = "multi_class"

otx_dir = os.getcwd()
templates = Registry("otx/algorithms/action").filter(task_type=TASK_TYPE.upper()).templates
templates_ids = [template.model_template_id for template in templates]


result_dict = get_result_dict(TASK_TYPE)
result_dir = f"/tmp/regression_test_results/{TASK_TYPE}"
Path(result_dir).mkdir(parents=True, exist_ok=True)

# Action Classification
action_det_regression_config = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, LABEL_TYPE)
action_det_data_args = action_det_regression_config["data_path"]
action_det_data_args["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]


class TestRegressionActionDetection:
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
        
    def teardown(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        self.performance[template.name] = {}
        
        tmp_dir_path = tmp_dir_path / TASK_TYPE
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, action_det_data_args)
        train_elapsed_time = timer() - train_start_time
        
        infer_start_time = timer()
        otx_eval_compare(
            template, tmp_dir_path, otx_dir, action_det_data_args, 
            action_det_regression_config["regression_criteria"]["train"], 
            self.performance[template.name],
            self.acc_metric
        )
        infer_elapsed_time = timer() - infer_start_time
        
        self.performance[template.name][self.train_time] = round(train_elapsed_time, 3)
        self.performance[template.name][self.infer_time] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][LABEL_TYPE][TRAIN_TYPE]["train"].append(self.performance)