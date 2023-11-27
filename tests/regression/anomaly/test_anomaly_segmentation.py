"""Tests for Anomaly Segmentation with OTX CLI."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import json
import os
import random
from pathlib import Path
from timeit import default_timer as timer

import pytest

from otx.cli.registry import Registry
from tests.regression.regression_command import (
    regression_deployment_testing,
    regression_eval_testing,
    regression_eval_time_testing,
    regression_nncf_eval_testing,
    regression_openvino_testing,
    regression_ptq_eval_testing,
    regression_train_time_testing,
)
from tests.regression.regression_test_helpers import (
    ANOMALY_DATASET_CATEGORIES,
    TIME_LOG,
    RegressionTestConfig,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_export_testing,
    otx_train_testing,
    ptq_optimize_testing,
)


class TestRegressionAnomalySegmentation:
    # Configurations for regression test.
    REG_CATEGORY = "anomaly"
    TASK_TYPE = "anomaly_segmentation"
    TRAIN_TYPE = None
    LABEL_TYPE = None
    TRAIN_PARAMS = None

    SAMPLED_ANOMALY_DATASET_CATEGORIES = ANOMALY_DATASET_CATEGORIES

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
            enable_auto_num_worker=False,
            results_root=results_root,
        )

        yield cls.reg_cfg

        cls.reg_cfg.dump_result_dict(dump_path=os.path.join(cls.reg_cfg.result_dir, f"result_{cls.TASK_TYPE}.json"))

    def setup_method(self):
        self.performance = {}

    def _apply_category(self, data_dict, category):
        return_dict = {}
        for k, v in data_dict.items():
            if "train" in k:
                return_dict[k] = f"{v}/{category}/train"
            if "val" in k or "test" in k:
                return_dict[k] = f"{v}/{category}/test"
            else:
                return_dict[k] = v
        return return_dict

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("category", SAMPLED_ANOMALY_DATASET_CATEGORIES)
    def test_otx_train(self, reg_cfg, template, tmp_dir_path, category):
        test_type = "train"
        self.performance[template.name] = {}
        category_data_args = self._apply_category(reg_cfg.args, category)

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        train_start_time = timer()
        otx_train_testing(template, tmp_dir_path, reg_cfg.otx_dir, category_data_args, deterministic=False)
        train_elapsed_time = timer() - train_start_time

        infer_start_time = timer()
        test_result = regression_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            category_data_args,
            reg_cfg.config_dict["regression_criteria"][test_type][category],
            self.performance[template.name],
        )
        infer_elapsed_time = timer() - infer_start_time

        self.performance[template.name][TIME_LOG["train_time"]] = round(train_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["infer_time"]] = round(infer_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance, is_anomaly=True, category=category)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("category", SAMPLED_ANOMALY_DATASET_CATEGORIES)
    def test_otx_train_kpi_test(self, reg_cfg, template, category):
        """KPI tests: measure the train+val time and evaluation time and compare with criteria."""
        performance = reg_cfg.get_template_performance(template, category=category)
        if performance is None:
            pytest.skip(reason="Cannot find performance data from results.")

        # Compare train+val time with the KPI criteria.
        kpi_train_result = regression_train_time_testing(
            train_time_criteria=reg_cfg.config_dict["kpi_e2e_train_time_criteria"]["train"][category],
            e2e_train_time=performance[template.name][TIME_LOG["train_time"]],
            template=template,
        )

        # Compare evaluation time with the KPI criteria.
        kpi_eval_result = regression_eval_time_testing(
            eval_time_criteria=reg_cfg.config_dict["kpi_e2e_eval_time_criteria"]["train"][category],
            e2e_eval_time=performance[template.name][TIME_LOG["infer_time"]],
            template=template,
        )

        assert kpi_train_result["passed"] is True, kpi_train_result["log"]
        assert kpi_eval_result["passed"] is True, kpi_eval_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("category", SAMPLED_ANOMALY_DATASET_CATEGORIES)
    def test_otx_export_eval_openvino(self, reg_cfg, template, tmp_dir_path, category):
        if category in ["metal_nut", "screw"]:
            pytest.skip("Issue#2189: Anomaly task sometimes shows performance drop")
        test_type = "export"
        self.performance[template.name] = {}
        category_data_args = self._apply_category(reg_cfg.args, category)

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        export_start_time = timer()
        otx_export_testing(template, tmp_dir_path)
        export_elapsed_time = timer() - export_start_time

        export_eval_start_time = timer()
        test_result = regression_openvino_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            category_data_args,
            threshold=0.05,
            criteria=reg_cfg.config_dict["regression_criteria"][test_type][category],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        export_eval_elapsed_time = timer() - export_eval_start_time

        self.performance[template.name][TIME_LOG["export_time"]] = round(export_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["export_eval_time"]] = round(export_eval_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance, is_anomaly=True, category=category)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("category", SAMPLED_ANOMALY_DATASET_CATEGORIES)
    def test_otx_deploy_eval_deployment(self, reg_cfg, template, tmp_dir_path, category):
        if category in ["metal_nut", "screw"]:
            pytest.skip("Issue#2189: Anomaly task sometimes shows performance drop")
        test_type = "deploy"
        self.performance[template.name] = {}
        category_data_args = self._apply_category(reg_cfg.args, category)

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        deploy_start_time = timer()
        otx_deploy_openvino_testing(template, tmp_dir_path, reg_cfg.otx_dir, category_data_args)
        deploy_elapsed_time = timer() - deploy_start_time

        deploy_eval_start_time = timer()
        test_result = regression_deployment_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            category_data_args,
            threshold=0.0,
            criteria=reg_cfg.config_dict["regression_criteria"][test_type][category],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        deploy_eval_elapsed_time = timer() - deploy_eval_start_time

        self.performance[template.name][TIME_LOG["deploy_time"]] = round(deploy_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["deploy_eval_time"]] = round(deploy_eval_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance, is_anomaly=True, category=category)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("category", SAMPLED_ANOMALY_DATASET_CATEGORIES)
    def test_nncf_optimize_eval(self, reg_cfg, template, tmp_dir_path, category):
        if category in ["screw"]:
            pytest.skip("Issue#2189: Anomaly task sometimes shows performance drop")
        test_type = "nncf"
        self.performance[template.name] = {}
        category_data_args = self._apply_category(reg_cfg.args, category)

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_start_time = timer()
        nncf_optimize_testing(template, tmp_dir_path, reg_cfg.otx_dir, category_data_args)
        nncf_elapsed_time = timer() - nncf_start_time

        nncf_eval_start_time = timer()
        test_result = regression_nncf_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            category_data_args,
            threshold=0.01,
            criteria=reg_cfg.config_dict["regression_criteria"][test_type][category],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        nncf_eval_elapsed_time = timer() - nncf_eval_start_time

        self.performance[template.name][TIME_LOG["nncf_time"]] = round(nncf_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["nncf_eval_time"]] = round(nncf_eval_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance, is_anomaly=True, category=category)

        assert test_result["passed"] is True, test_result["log"]

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.parametrize("category", SAMPLED_ANOMALY_DATASET_CATEGORIES)
    def test_ptq_optimize_eval(self, reg_cfg, template, tmp_dir_path, category):
        if category in ["metal_nut", "screw"]:
            pytest.skip("Issue#2189: Anomaly task sometimes shows performance drop")
        test_type = "ptq"
        self.performance[template.name] = {}
        category_data_args = self._apply_category(reg_cfg.args, category)

        tmp_dir_path = tmp_dir_path / reg_cfg.task_type
        ptq_start_time = timer()
        ptq_optimize_testing(template, tmp_dir_path, reg_cfg.otx_dir, category_data_args)
        ptq_elapsed_time = timer() - ptq_start_time

        ptq_eval_start_time = timer()
        test_result = regression_ptq_eval_testing(
            template,
            tmp_dir_path,
            reg_cfg.otx_dir,
            category_data_args,
            criteria=reg_cfg.config_dict["regression_criteria"][test_type][category],
            reg_threshold=0.10,
            result_dict=self.performance[template.name],
        )
        ptq_eval_elapsed_time = timer() - ptq_eval_start_time

        self.performance[template.name][TIME_LOG["ptq_time"]] = round(ptq_elapsed_time, 3)
        self.performance[template.name][TIME_LOG["ptq_eval_time"]] = round(ptq_eval_elapsed_time, 3)
        reg_cfg.update_result(test_type, self.performance, is_anomaly=True, category=category)

        assert test_result["passed"] is True, test_result["log"]
