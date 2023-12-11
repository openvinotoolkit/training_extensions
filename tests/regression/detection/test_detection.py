"""Tests for Detection with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import json
import os
from pathlib import Path
from timeit import default_timer as timer

import pytest

from otx.cli.registry import Registry
from tests.regression.regression_test_helpers import (
    REGRESSION_TEST_EPOCHS,
    TIME_LOG,
    RegressionTestConfig,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    get_template_dir,
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_export_testing,
    otx_train_testing,
    ptq_optimize_testing,
)

from tests.regression.regression_command import (
    regression_eval_testing,
    regression_openvino_testing,
    regression_deployment_testing,
    regression_nncf_eval_testing,
    regression_ptq_eval_testing,
    regression_train_time_testing,
    regression_eval_time_testing,
)


class TestRegressionDetection:
    REG_CATEGORY = "detection"
    TASK_TYPE = "detection"
    TRAIN_TYPE = "supervised"
    LABEL_TYPE = "multi_class"

    TRAIN_PARAMS = ["--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]

    templates = Registry(f"src/otx/algorithms/{REG_CATEGORY}").filter(task_type=TASK_TYPE.upper()).templates
    templates_ids = [template.model_template_id for template in templates]

    reg_cfg: RegressionTestConfig

    @classmethod
    @pytest.fixture(scope="class")
    def reg_cfg(cls, tmp_dir_path):
        results_root = os.environ.get("REG_RESULTS_ROOT", tmp_dir_path)
        cls.reg_cfg = RegressionTestConfig(
            cls.TASK_TYPE,
            cls.TRAIN_TYPE,
            cls.LABEL_TYPE,
            os.getcwd(),
            train_params=cls.TRAIN_PARAMS,
            results_root=results_root,
        )

        yield cls.reg_cfg

        cls.reg_cfg.dump_result_dict()

    def setup_method(self):
        self.performance = {}

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, reg_cfg, template, tmp_dir_path):
        test_type = "train"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, reg_cfg.otx_dir, reg_cfg.args)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            reg_cfg.args,
            reg_cfg.config_dict["regression_criteria"][test_type],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_kpi_test(self, reg_cfg, template):
        performance = reg_cfg.get_template_performance(template)
        if performance is None:
            pytest.skip(reason="Cannot find performance data from results.")

        kpi_train_result = regression_train_time_testing(
            train_time_criteria=reg_cfg.config_dict["kpi_e2e_train_time_criteria"]["train"],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=reg_cfg.config_dict["kpi_e2e_eval_time_criteria"]["train"],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_cls_incr(self, reg_cfg, template, tmp_dir_path):
        train_type = "class_incr"
        test_type = "train"
        self.performance[template.name] = {}

        sl_template_work_dir = get_template_dir(template, tmp_dir_path / reg_cfg.task_type)

        tmp_dir_path = tmp_dir_path / "det_incr"
        config_cls_incr = reg_cfg.load_config(train_type=train_type)
        args_cls_incr = config_cls_incr["data_path"]
        args_cls_incr[
            "--load-weights"
        ] = f"{sl_template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        args_cls_incr["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]

        reg_cfg.update_gpu_args(args_cls_incr)

        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, reg_cfg.otx_dir, args_cls_incr)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            args_cls_incr,
            config_cls_incr["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance, train_type=train_type)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_cls_incr_kpi_test(self, reg_cfg, template):
        train_type = "class_incr"
        config_cls_incr = reg_cfg.load_config(train_type=train_type)
        performance = reg_cfg.get_template_performance(template, train_type=train_type)
        if performance is None:
            pytest.skip(reason="Cannot find performance data from results.")

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
    def test_otx_train_semisl(self, reg_cfg, template, tmp_dir_path):
        train_type = "semi_supervised"
        test_type = "train"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / f"{reg_cfg.task_type}/test_semisl"
        config_semisl = reg_cfg.load_config(train_type=train_type)
        args_semisl = config_semisl["data_path"]

        args_semisl["train_params"] = [
            "params",
            "--learning_parameters.num_iters",
            REGRESSION_TEST_EPOCHS,
            "--algo_backend.train_type",
            "Semisupervised",
        ]
        reg_cfg.update_gpu_args(args_semisl)

        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, reg_cfg.otx_dir, args_semisl)
        train_elapsed_time = timer() - train_start_time

        args_semisl.pop("train_params")
        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            args_semisl,
            config_semisl["regression_criteria"]["train"],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance, train_type=train_type)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_semisl_kpi_test(self, reg_cfg, template):
        train_type = "semi_supervised"
        config_semisl = reg_cfg.load_config(train_type=train_type)
        performance = reg_cfg.get_template_performance(template, train_type=train_type)
        if performance is None:
            pytest.skip(reason="Cannot find performance data from results.")

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
    def test_otx_export_eval_openvino(self, reg_cfg, template, tmp_dir_path):
        test_type = "export"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        export_start_time = timer()
        otx_export_testing(template, tmp_dir_path)
        export_elapsed_time = timer() - export_start_time

        export_eval_start_time = timer()
        test_result = regression_openvino_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            reg_cfg.args,
            threshold=0.05,
            criteria=reg_cfg.config_dict["regression_criteria"][test_type],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        export_eval_elapsed_time = timer() - export_eval_start_time

        self.performance[template.name][TIME_LOG["export_time"]] = round(export_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["export_eval_time"]] = round(export_eval_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_eval_deployment(self, reg_cfg, template, tmp_dir_path):
        test_type = "deploy"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        deploy_start_time = timer()
        otx_deploy_openvino_testing(template, tmp_dir_path, reg_cfg.otx_dir, reg_cfg.args)
        deploy_elapsed_time = timer() - deploy_start_time

        deploy_eval_start_time = timer()
        test_result = regression_deployment_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            reg_cfg.args,
            threshold=0.0,
            criteria=reg_cfg.config_dict["regression_criteria"][test_type],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        deploy_eval_elapsed_time = timer() - deploy_eval_start_time

        self.performance[template.name][TIME_LOG["deploy_time"]] = round(deploy_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["deploy_eval_time"]] = round(deploy_eval_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize_eval(self, reg_cfg, template, tmp_dir_path):
        if template.name == "YOLOX-S":
            pytest.skip("Issue#2596: IndexError")
        test_type = "nncf"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_start_time = timer()
        nncf_optimize_testing(template, tmp_dir_path, reg_cfg.otx_dir, reg_cfg.args)
        nncf_elapsed_time = timer() - nncf_start_time

        nncf_eval_start_time = timer()
        test_result = regression_nncf_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            reg_cfg.args,
            threshold=0.01,
            criteria=reg_cfg.config_dict["regression_criteria"][test_type],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        nncf_eval_elapsed_time = timer() - nncf_eval_start_time

        self.performance[template.name][TIME_LOG["nncf_time"]] = round(nncf_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["nncf_eval_time"]] = round(nncf_eval_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_optimize_eval(self, reg_cfg, template, tmp_dir_path):
        test_type = "ptq"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        ptq_start_time = timer()
        ptq_optimize_testing(template, tmp_dir_path, reg_cfg.otx_dir, reg_cfg.args)
        ptq_elapsed_time = timer() - ptq_start_time

        ptq_eval_start_time = timer()
        test_result = regression_ptq_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            reg_cfg.args,
            criteria=reg_cfg.config_dict["regression_criteria"][test_type],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        ptq_eval_elapsed_time = timer() - ptq_eval_start_time

        self.performance[template.name][TIME_LOG["ptq_time"]] = round(ptq_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["ptq_eval_time"]] = round(ptq_eval_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance)

        assert test_result["passed"] is True, test_result["log"]
