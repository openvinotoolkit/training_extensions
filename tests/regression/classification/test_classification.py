"""Tests for Classification with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import json
import os
from pathlib import Path

import pytest

from otx.cli.registry import Registry
from tests.regression.regression_test_helpers import (
    get_result_dict,
    load_regression_configuration,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    get_template_dir,
    nncf_eval_testing,
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_eval_compare,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_export_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)

otx_dir = os.getcwd()
templates = Registry("otx/algorithms/classification").filter(task_type="CLASSIFICATION").templates
templates_ids = [template.model_template_id for template in templates]

# Configurations for regression test.
REGRESSION_TEST_EPOCHS = "1"
TASK_TYPE = "classification"

result_dict = get_result_dict(TASK_TYPE)
result_dir = f"/tmp/regression_test_results_{TASK_TYPE}"
Path(result_dir).mkdir(parents=True, exist_ok=True)

# Multi-class Classification
multi_class_regression_config = load_regression_configuration(otx_dir, "classification", "supervised", "multi_class")
multi_class_data_args = multi_class_regression_config["data_path"]
multi_class_data_args["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]


class TestRegressionMultiClassClassification:
    def setup(self):
        self.label_type = "multi_class"

    def teardown(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)
        otx_eval_compare(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            multi_class_regression_config["regression_criteria"],
            result_dict["train"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_semisl"
        config_semisl = load_regression_configuration(otx_dir, "classification", "semi_supervised", "multi_class")
        args_semisl = config_semisl["data_path"]

        args_semisl["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]
        args_semisl["train_params"].extend(["--algo_backend.train_type", "SEMISUPERVISED"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)

        args_semisl.pop("train_params")
        otx_eval_compare(
            template,
            tmp_dir_path,
            otx_dir,
            args_semisl,
            config_semisl["regression_criteria"],
            result_dict["train"][TASK_TYPE][self.label_type]["semi_supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_selfsl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_selfsl"
        config_selfsl = load_regression_configuration(otx_dir, "classification", "self_supervised", "multi_class")
        args_selfsl = config_selfsl["data_path"]

        args_selfsl["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]
        selfsl_train_args = copy.deepcopy(args_selfsl)
        selfsl_train_args["train_params"].extend(["--algo_backend.train_type", "SELFSUPERVISED"])
        # Self-supervised Training
        otx_train_testing(template, tmp_dir_path, otx_dir, selfsl_train_args)

        # Supervised Training
        template_work_dir = get_template_dir(template, tmp_dir_path)

        new_tmp_dir_path = tmp_dir_path / "test_supervised"
        args_selfsl["--val-data-roots"] = "/storageserver/pvd_data/otx_data_archive/classification/cifar10_subset/test"
        args_selfsl["--test-data-roots"] = "/storageserver/pvd_data/otx_data_archive/classification/cifar10_subset/test"
        args_selfsl["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        otx_train_testing(template, new_tmp_dir_path, otx_dir, args_selfsl)

        # Evaluation with self + supervised training model
        otx_eval_compare(
            template,
            new_tmp_dir_path,
            otx_dir,
            args_selfsl,
            config_selfsl["regression_criteria"],
            result_dict["train"][TASK_TYPE][self.label_type]["self_supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            threshold=0.0,
            result_dict=result_dict["export"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_deployment_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            threshold=0.0,
            result_dict=result_dict["deploy"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            threshold=0.001,
            result_dict=result_dict["nncf"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        pot_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_class_data_args,
            result_dict=result_dict["pot"][TASK_TYPE][self.label_type]["supervised"],
        )


# Multi-label Classification
multi_label_regression_config = load_regression_configuration(otx_dir, "classification", "supervised", "multi_label")
multi_label_data_args = multi_label_regression_config["data_path"]
multi_label_data_args["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]


class TestRegressionMultiLabelClassification:
    def setup(self):
        self.label_type = "multi_label"

    def teardown(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)
        otx_eval_compare(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            multi_label_regression_config["regression_criteria"],
            result_dict["train"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_eval_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            threshold=0.0,
            result_dict=result_dict["export"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_eval_deployment_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            threshold=0.0,
            result_dict=result_dict["deploy"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            threshold=0.001,
            result_dict=result_dict["nncf"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        pot_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            multi_label_data_args,
            result_dict=result_dict["pot"][TASK_TYPE][self.label_type]["supervised"],
        )


# H-label Classification
h_label_regression_config = load_regression_configuration(otx_dir, "classification", "supervised", "h_label")
h_label_data_args = h_label_regression_config["data_path"]
h_label_data_args["train_params"] = ["params", "--learning_parameters.num_iters", REGRESSION_TEST_EPOCHS]


class TestRegressionHierarchicalLabelClassification:
    def setup(self):
        self.label_type = "h_label"

    def teardown(self):
        with open(f"{result_dir}/result.json", "w") as result_file:
            json.dump(result_dict, result_file, indent=4)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, h_label_data_args)
        otx_eval_compare(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            h_label_regression_config["regression_criteria"],
            result_dict["train"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_eval_openvino_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            threshold=0.0,
            result_dict=result_dict["export"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, h_label_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_eval_deployment_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            threshold=0.0,
            result_dict=result_dict["deploy"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, h_label_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            threshold=0.001,
            result_dict=result_dict["nncf"][TASK_TYPE][self.label_type]["supervised"],
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, h_label_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        pot_eval_testing(
            template,
            tmp_dir_path,
            otx_dir,
            h_label_data_args,
            result_dict=result_dict["pot"][TASK_TYPE][self.label_type]["supervised"],
        )
