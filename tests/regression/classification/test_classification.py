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

# Configurations for regression test.
TASK_TYPE = "classification"
TRAIN_TYPE = "supervised"

otx_dir = os.getcwd()
templates = Registry("src/otx/algorithms/classification").filter(task_type=TASK_TYPE.upper()).templates
templates_ids = [template.model_template_id for template in templates]

result_dict = get_result_dict(TASK_TYPE)
result_dir = f"/tmp/regression_test_results/{TASK_TYPE}"
Path(result_dir).mkdir(parents=True, exist_ok=True)

multi_class_regression_config = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, "multi_class")
multi_class_data_args = multi_class_regression_config["data_path"]
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
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            multi_class_regression_config["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_kpi_test(self, template):
        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=multi_class_regression_config["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=multi_class_regression_config["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

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
        args_cls_incr["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]

        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, args_cls_incr)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            args_cls_incr,
            config_cls_incr["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type]["class_incr"]["train"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_cls_incr_kpi_test(self, template):
        config_cls_incr = load_regression_configuration(otx_dir, TASK_TYPE, "class_incr", self.label_type)

        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=config_cls_incr["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=config_cls_incr["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_semisl(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_semisl"
        config_semisl = load_regression_configuration(otx_dir, TASK_TYPE, "semi_supervised", self.label_type)
        args_semisl = config_semisl["data_path"]

        args_semisl["train_params"] = [
            "params",
            "--learning_parameters.num_iters",
            REGRESSION_TEST_EPOCHS,
            "--algo_backend.train_type",
            "Semisupervised",
        ]
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            args_semisl,
            config_semisl["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type]["semi_supervised"]["train"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_semisl_kpi_test(self, template):
        config_semisl = load_regression_configuration(otx_dir, TASK_TYPE, "semi_supervised", self.label_type)

        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=config_semisl["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=config_semisl["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_selfsl(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_selfsl"
        config_selfsl = load_regression_configuration(otx_dir, TASK_TYPE, "self_supervised", self.label_type)
        args_selfsl = config_selfsl["data_path"]

        selfsl_train_args = copy.deepcopy(args_selfsl)
        selfsl_train_args["train_params"] = [
            "params",
            "--learning_parameters.batch_size",
            "64",
            "--learning_parameters.num_iters",
            "10",
            "--algo_backend.train_type",
            "Selfsupervised",
        ]

        # Self-supervised Training
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, selfsl_train_args)
        train_elapsed_time = timer() - train_start_time

        # Supervised Training
        template_work_dir = get_template_dir(template, tmp_dir_path)
        new_tmp_dir_path = tmp_dir_path / "test_supervised"
        args_selfsl["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]
        args_selfsl["--val-data-roots"] = multi_class_data_args["--val-data-roots"]
        args_selfsl["--test-data-roots"] = multi_class_data_args["--test-data-roots"]
        args_selfsl["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        otx_train_testing(template, new_tmp_dir_path, otx_dir, args_selfsl)

        # Evaluation with self + supervised training model
        args_selfsl.pop("--load-weights")
        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            new_tmp_dir_path,
            otx_dir,
            args_selfsl,
            config_selfsl["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type]["self_supervised"]["train"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_selfsl_kpi_test(self, template):
        config_selfsl = load_regression_configuration(otx_dir, TASK_TYPE, "self_supervised", self.label_type)

        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=config_selfsl["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=config_selfsl["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_eval_openvino(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        export_start_time = timer()
        otx_export_testing(template, tmp_dir_path)
        export_elapsed_time = timer() - export_start_time

        export_eval_start_time = timer()
        test_result = regression_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            threshold=0.05,
            criteria=multi_class_regression_config["regression_criteria"]["export"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        export_eval_elapsed_time = timer() - export_eval_start_time

        self.performance[template.name][TIME_LOG["export_time"]] = round(export_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["export_eval_time"]] = round(export_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["export"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_eval_deployment(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        deploy_start_time = timer()
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)
        deploy_elapsed_time = timer() - deploy_start_time

        deploy_eval_start_time = timer()
        test_result = regression_deployment_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            threshold=0.0,
            criteria=multi_class_regression_config["regression_criteria"]["deploy"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        deploy_eval_elapsed_time = timer() - deploy_eval_start_time

        self.performance[template.name][TIME_LOG["deploy_time"]] = round(deploy_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["deploy_eval_time"]] = round(deploy_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["deploy"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_start_time = timer()
        nncf_optimize_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)
        nncf_elapsed_time = timer() - nncf_start_time

        nncf_eval_start_time = timer()
        test_result = regression_nncf_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            threshold=0.01,
            criteria=multi_class_regression_config["regression_criteria"]["nncf"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        nncf_eval_elapsed_time = timer() - nncf_eval_start_time

        self.performance[template.name][TIME_LOG["nncf_time"]] = round(nncf_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["nncf_eval_time"]] = round(nncf_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["nncf"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        pot_start_time = timer()
        pot_optimize_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)
        pot_elapsed_time = timer() - pot_start_time

        pot_eval_start_time = timer()
        test_result = regression_pot_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            criteria=multi_class_regression_config["regression_criteria"]["pot"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        pot_eval_elapsed_time = timer() - pot_eval_start_time

        self.performance[template.name][TIME_LOG["pot_time"]] = round(pot_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["pot_eval_time"]] = round(pot_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["pot"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]


multi_label_regression_config = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, "multi_label")
multi_label_data_args = multi_label_regression_config["data_path"]
multi_label_data_args["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]


class TestRegressionMultiLabelClassification:
    def setup_method(self):
        self.label_type = "multi_label"
        self.performance = {}

    def teardown_method(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            multi_label_regression_config["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_kpi_test(self, template):
        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=multi_label_regression_config["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=multi_label_regression_config["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_cls_incr(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        sl_template_work_dir = get_template_dir(template, tmp_dir_path / "multi_label_cls")

        tmp_dir_path = tmp_dir_path / "multi_label_cls_incr"
        config_cls_incr = load_regression_configuration(otx_dir, TASK_TYPE, "class_incr", self.label_type)
        args_cls_incr = config_cls_incr["data_path"]
        args_cls_incr[
            "--load-weights"
        ] = f"{sl_template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        args_cls_incr["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]

        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, args_cls_incr)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            args_cls_incr,
            config_cls_incr["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type]["class_incr"]["train"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_cls_incr_kpi_test(self, template):
        config_cls_incr = load_regression_configuration(otx_dir, TASK_TYPE, "class_incr", self.label_type)

        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=config_cls_incr["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=config_cls_incr["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_eval_openvino(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        export_start_time = timer()
        otx_export_testing(template, tmp_dir_path)
        export_elapsed_time = timer() - export_start_time

        export_eval_start_time = timer()
        test_result = regression_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            threshold=0.05,
            criteria=multi_label_regression_config["regression_criteria"]["export"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        export_eval_elapsed_time = timer() - export_eval_start_time

        self.performance[template.name][TIME_LOG["export_time"]] = round(export_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["export_eval_time"]] = round(export_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["export"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_eval_deployment(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        deploy_start_time = timer()
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)
        deploy_elapsed_time = timer() - deploy_start_time

        deploy_eval_start_time = timer()
        test_result = regression_deployment_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            threshold=0.0,
            criteria=multi_label_regression_config["regression_criteria"]["deploy"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        deploy_eval_elapsed_time = timer() - deploy_eval_start_time

        self.performance[template.name][TIME_LOG["deploy_time"]] = round(deploy_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["deploy_eval_time"]] = round(deploy_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["deploy"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_start_time = timer()
        nncf_optimize_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)
        nncf_elapsed_time = timer() - nncf_start_time

        nncf_eval_start_time = timer()
        test_result = regression_nncf_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            threshold=0.01,
            criteria=multi_label_regression_config["regression_criteria"]["nncf"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        nncf_eval_elapsed_time = timer() - nncf_eval_start_time

        self.performance[template.name][TIME_LOG["nncf_time"]] = round(nncf_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["nncf_eval_time"]] = round(nncf_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["nncf"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        pot_start_time = timer()
        pot_optimize_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)
        pot_elapsed_time = timer() - pot_start_time

        pot_eval_start_time = timer()
        test_result = regression_pot_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            criteria=multi_label_regression_config["regression_criteria"]["pot"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        pot_eval_elapsed_time = timer() - pot_eval_start_time

        self.performance[template.name][TIME_LOG["pot_time"]] = round(pot_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["pot_eval_time"]] = round(pot_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["pot"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]


h_label_regression_config = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, "h_label")
h_label_data_args = h_label_regression_config["data_path"]
h_label_data_args["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]


class TestRegressionHierarchicalLabelClassification:
    def setup_method(self):
        self.label_type = "h_label"
        self.performance = {}

    def teardown_method(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "h_label_cls"
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, h_label_data_args)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            h_label_regression_config["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_kpi_test(self, template):
        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=h_label_regression_config["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=h_label_regression_config["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export_eval_openvino(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "h_label_cls"
        export_start_time = timer()
        otx_export_testing(template, tmp_dir_path)
        export_elapsed_time = timer() - export_start_time

        export_eval_start_time = timer()
        test_result = regression_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            threshold=0.05,
            criteria=h_label_regression_config["regression_criteria"]["export"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        export_eval_elapsed_time = timer() - export_eval_start_time

        self.performance[template.name][TIME_LOG["export_time"]] = round(export_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["export_eval_time"]] = round(export_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["export"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_eval_deployment(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "h_label_cls"
        deploy_start_time = timer()
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, h_label_data_args)
        deploy_elapsed_time = timer() - deploy_start_time

        deploy_eval_start_time = timer()
        test_result = regression_deployment_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            threshold=0.0,
            criteria=h_label_regression_config["regression_criteria"]["deploy"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        deploy_eval_elapsed_time = timer() - deploy_eval_start_time

        self.performance[template.name][TIME_LOG["deploy_time"]] = round(deploy_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["deploy_eval_time"]] = round(deploy_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["deploy"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_start_time = timer()
        nncf_optimize_testing(template, tmp_dir_path, otx_dir, h_label_data_args)
        nncf_elapsed_time = timer() - nncf_start_time

        nncf_eval_start_time = timer()
        test_result = regression_nncf_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            threshold=0.01,
            criteria=h_label_regression_config["regression_criteria"]["nncf"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        nncf_eval_elapsed_time = timer() - nncf_eval_start_time

        self.performance[template.name][TIME_LOG["nncf_time"]] = round(nncf_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["nncf_eval_time"]] = round(nncf_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["nncf"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize_eval(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "h_label_cls"
        pot_start_time = timer()
        pot_optimize_testing(template, tmp_dir_path, otx_dir, h_label_data_args)
        pot_elapsed_time = timer() - pot_start_time

        pot_eval_start_time = timer()
        test_result = regression_pot_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            criteria=h_label_regression_config["regression_criteria"]["pot"],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        pot_eval_elapsed_time = timer() - pot_eval_start_time

        self.performance[template.name][TIME_LOG["pot_time"]] = round(pot_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["pot_eval_time"]] = round(pot_eval_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["pot"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]


class TestRegressionSupconClassification:
    def setup_method(self):
        self.label_type = "supcon"
        self.performance = {}

    def teardown_method(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "supcon_cls"
        config_supcon = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, self.label_type)
        args_supcon = config_supcon["data_path"]

        args_supcon["train_params"] = [
            "params",
            "--learning_parameters.num_iters",
            REGRESSION_TEST_EPOCHS,
            "--learning_parameters.enable_supcon",
            "True",
        ]
        # Supcon
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, otx_dir, args_supcon)
        train_elapsed_time = timer() - train_start_time

        # Evaluation with supcon + supervised training
        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            args_supcon,
            config_supcon["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"].append(self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_kpi_test(self, template):
        config_supcon = load_regression_configuration(otx_dir, TASK_TYPE, TRAIN_TYPE, self.label_type)
        results = result_dict[TASK_TYPE][self.label_type][TRAIN_TYPE]["train"]
        performance = get_template_performance(results, template)

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=config_supcon["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=config_supcon["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]
