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


class TestRegressionMultiClassClassification:
    REG_CATEGORY = "classification"
    TASK_TYPE = "classification"
    TRAIN_TYPE = "supervised"
    LABEL_TYPE = "multi_class"

    TRAIN_PARAMS = ["--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]
    templates = Registry(f"src/otx/algorithms/{REG_CATEGORY}").filter(task_type=TASK_TYPE.upper()).templates
    templates_ids = [template.model_template_id for template in templates]

    reg_cfg: RegressionTestConfig

    @classmethod
    @pytest.fixture(scope="class")
    def reg_cfg(cls):
        cls.reg_cfg = RegressionTestConfig(
            cls.TASK_TYPE,
            cls.TRAIN_TYPE,
            cls.LABEL_TYPE,
            os.getcwd(),
            train_params=cls.TRAIN_PARAMS,
        )

        yield cls.reg_cfg

        print(f"\nwritting regression result to {cls.reg_cfg.result_dir}/result_{cls.TRAIN_TYPE}_{cls.LABEL_TYPE}.json")
        with open(f"{cls.reg_cfg.result_dir}/result_{cls.TRAIN_TYPE}_{cls.LABEL_TYPE}.json", "w") as result_file:
            json.dump(cls.reg_cfg.result_dict, result_file, indent=4)

    def setup_method(self):
        self.performance = {}

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, reg_cfg, template, tmp_dir_path):
        test_type = "train"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
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

        sl_template_work_dir = get_template_dir(template, tmp_dir_path / "multi_class_cls")

        tmp_dir_path = tmp_dir_path / "multi_class_cls_incr"
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
            config_cls_incr["regression_criteria"][test_type],
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

        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_semisl"
        # config_semisl = load_regression_configuration(reg_cfg.otx_dir, TASK_TYPE, "semi_supervised", reg_cfg.label_type)
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

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            args_semisl,
            config_semisl["regression_criteria"][test_type],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance)

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
    def test_otx_train_selfsl(self, reg_cfg, template, tmp_dir_path):
        if template.name == "DeiT-Tiny":
            pytest.skip(reason="Self-SL for ViT template is not supported yet.")
        train_type = "self_supervised"
        test_type = "train"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_selfsl"
        config_selfsl = reg_cfg.load_config(train_type=train_type)
        args_selfsl = config_selfsl["data_path"]

        selfsl_train_args = copy.deepcopy(args_selfsl)
        selfsl_train_args["--train-type"] = "Selfsupervised"
        selfsl_train_args["train_params"] = [
            "params",
            "--learning_parameters.batch_size",
            "64",
            "--learning_parameters.num_iters",
            "10",
        ]

        reg_cfg.update_gpu_args(selfsl_train_args)

        # Self-supervised Training
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, reg_cfg.otx_dir, selfsl_train_args)
        train_elapsed_time = timer() - train_start_time

        # Supervised Training
        template_work_dir = get_template_dir(template, tmp_dir_path)
        assert os.path.exists(f"{template_work_dir}/selfsl")
        new_tmp_dir_path = tmp_dir_path / "test_supervised"
        args_selfsl["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]
        args_selfsl["--val-data-roots"] = reg_cfg.args["--val-data-roots"]
        args_selfsl["--test-data-roots"] = reg_cfg.args["--test-data-roots"]
        args_selfsl["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"

        reg_cfg.update_gpu_args(args_selfsl)

        otx_train_testing(template, new_tmp_dir_path, reg_cfg.otx_dir, args_selfsl)

        # Evaluation with self + supervised training model
        args_selfsl.pop("--load-weights")
        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            new_tmp_dir_path,
            reg_cfg.otx_dir,
            args_selfsl,
            config_selfsl["regression_criteria"][test_type],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_selfsl_kpi_test(self, reg_cfg, template):
        train_type = "self_supervised"
        config_selfsl = reg_cfg.load_config(train_type=train_type)

        performance = reg_cfg.get_template_performance(template, train_type=train_type)
        if performance is None:
            pytest.skip(reason="Cannot find performance data from results.")

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
    def test_otx_export_eval_openvino(self, reg_cfg, template, tmp_dir_path):
        test_type = "export"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
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

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
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
        test_type = "nncf"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
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

        tmp_dir_path = tmp_dir_path / "multi_class_cls"
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


class TestRegressionMultiLabelClassification:
    REG_CATEGORY = "classification"
    TASK_TYPE = "classification"
    TRAIN_TYPE = "supervised"
    LABEL_TYPE = "multi_label"

    TRAIN_PARAMS = ["--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]
    templates = Registry(f"src/otx/algorithms/{REG_CATEGORY}").filter(task_type=TASK_TYPE.upper()).templates
    templates_ids = [template.model_template_id for template in templates]

    reg_cfg: RegressionTestConfig

    @classmethod
    @pytest.fixture(scope="class")
    def reg_cfg(cls):
        cls.reg_cfg = RegressionTestConfig(
            cls.TASK_TYPE,
            cls.TRAIN_TYPE,
            cls.LABEL_TYPE,
            os.getcwd(),
            train_params=cls.TRAIN_PARAMS,
        )

        yield cls.reg_cfg

        print(f"\nwritting regression result to {cls.reg_cfg.result_dir}/result_{cls.TRAIN_TYPE}_{cls.LABEL_TYPE}.json")
        with open(f"{cls.reg_cfg.result_dir}/result_{cls.TRAIN_TYPE}_{cls.LABEL_TYPE}.json", "w") as result_file:
            json.dump(cls.reg_cfg.result_dict, result_file, indent=4)

    def setup_method(self):
        self.performance = {}

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, reg_cfg, template, tmp_dir_path):
        test_type = "train"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
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

        sl_template_work_dir = get_template_dir(template, tmp_dir_path / "multi_label_cls")

        tmp_dir_path = tmp_dir_path / "multi_label_cls_incr"
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
            config_cls_incr["regression_criteria"][test_type],
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
    def test_otx_export_eval_openvino(self, reg_cfg, template, tmp_dir_path):
        test_type = "export"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
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

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
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
        test_type = "nncf"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
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

        tmp_dir_path = tmp_dir_path / "multi_label_cls"
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


class TestRegressionHierarchicalLabelClassification:
    REG_CATEGORY = "classification"
    TASK_TYPE = "classification"
    TRAIN_TYPE = "supervised"
    LABEL_TYPE = "h_label"

    TRAIN_PARAMS = ["--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]
    templates = Registry(f"src/otx/algorithms/{REG_CATEGORY}").filter(task_type=TASK_TYPE.upper()).templates
    templates_ids = [template.model_template_id for template in templates]

    reg_cfg: RegressionTestConfig

    @classmethod
    @pytest.fixture(scope="class")
    def reg_cfg(cls):
        cls.reg_cfg = RegressionTestConfig(
            cls.TASK_TYPE,
            cls.TRAIN_TYPE,
            cls.LABEL_TYPE,
            os.getcwd(),
            train_params=cls.TRAIN_PARAMS,
        )

        yield cls.reg_cfg

        print(f"\nwritting regression result to {cls.reg_cfg.result_dir}/result_{cls.TRAIN_TYPE}_{cls.LABEL_TYPE}.json")
        with open(f"{cls.reg_cfg.result_dir}/result_{cls.TRAIN_TYPE}_{cls.LABEL_TYPE}.json", "w") as result_file:
            json.dump(cls.reg_cfg.result_dict, result_file, indent=4)

    def setup_method(self):
        self.performance = {}

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, reg_cfg, template, tmp_dir_path):
        test_type = "train"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "h_label_cls"
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
    def test_otx_export_eval_openvino(self, reg_cfg, template, tmp_dir_path):
        test_type = "export"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "h_label_cls"
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

        tmp_dir_path = tmp_dir_path / "h_label_cls"
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
        test_type = "nncf"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "h_label_cls"
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

        tmp_dir_path = tmp_dir_path / "h_label_cls"
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


class TestRegressionSupconClassification:
    REG_CATEGORY = "classification"
    TASK_TYPE = "classification"
    TRAIN_TYPE = "supervised"
    LABEL_TYPE = "supcon"

    TRAIN_PARAMS = [
        "--learning_parameters.num_iters",
        REGRESSION_TEST_EPOCHS,
        "--learning_parameters.enable_supcon",
        "True",
    ]
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

        print(f"\nwritting regression result to {cls.reg_cfg.result_dir}/result_{cls.TRAIN_TYPE}_{cls.LABEL_TYPE}.json")
        with open(f"{cls.reg_cfg.result_dir}/result_{cls.TRAIN_TYPE}_{cls.LABEL_TYPE}.json", "w") as result_file:
            json.dump(cls.reg_cfg.result_dict, result_file, indent=4)

    def setup_method(self):
        self.performance = {}

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, reg_cfg, template, tmp_dir_path):
        test_type = "train"
        self.performance[template.name] = {}

        tmp_dir_path = tmp_dir_path / "supcon_cls"

        # Supcon
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, reg_cfg.otx_dir, reg_cfg.args)
        train_elapsed_time = timer() - train_start_time

        # Evaluation with supcon + supervised training
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
