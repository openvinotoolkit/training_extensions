"""Tests for Classification with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import json
import os
from pathlib import Path
from timeit import default_timer as timer

import pytest

from otx.cli.registry import Registry
from tests.regression.regression_test_helpers import (
    REGRESSION_TEST_EPOCHS,
    TIME_LOG,
    get_result_dict,
    get_template_performance,
    load_regression_configuration,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    get_template_dir,
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_export_testing,
    otx_train_testing,
    pot_optimize_testing,
)

from tests.regression.regression_command import (
    regression_eval_testing,
    regression_openvino_testing,
    regression_deployment_testing,
    regression_nncf_eval_testing,
    regression_pot_eval_testing,
    regression_train_time_testing,
    regression_eval_time_testing,
)

REGRESSION_TEST_EPOCHS = "1"
# Configurations for regression test.
TASK_TYPE = "classification"
TRAIN_TYPE = "supervised"

otx_dir = os.getcwd()
templates = Registry("otx/algorithms/classification").filter(task_type=TASK_TYPE.upper()).templates
templates_ids = [template.model_template_id for template in templates]
templates = [templates[0]]
templates_ids = [templates_ids[0]]

result_dict = get_result_dict(TASK_TYPE)
result_dir = f"/tmp/regression_test_results/{TASK_TYPE}"
Path(result_dir).mkdir(parents=True, exist_ok=True)

multi_class_regression_config = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, "multi_class")
multi_class_data_args = multi_class_regression_config["data_path"]
multi_class_data_args["--gpus"] = "0,1"
multi_class_data_args["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]


class TestRegressionMultiClassClassification:
    def setup_method(self):
        self.label_type = "multi_class"
        self.performance = {}

    def teardown_method(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_cls_incr(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        sl_template_work_dir = get_template_dir(template, tmp_dir_path / "multi_class_cls")

        tmp_dir_path = tmp_dir_path / "multi_class_cls_incr"
        config_cls_incr = load_regression_configuration(otx_dir, TASK_TYPE, "class_incr", self.label_type)
        args_cls_incr = config_cls_incr["data_path"]
        args_cls_incr[
            "--load-weights"
        ] = f"{sl_template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        args_cls_incr["--gpus"] = "0,1"
        args_cls_incr["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]

        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, args_cls_incr)
