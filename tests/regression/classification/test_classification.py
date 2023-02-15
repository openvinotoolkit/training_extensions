"""Tests for Classification with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import os

import pytest

from otx.cli.registry import Registry
from otx.cli.utils.tests import (
    nncf_eval_testing,
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_regression_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)
from tests.regression.regression_test_helpers import load_regression_configuration
from tests.test_suite.e2e_test_system import e2e_pytest_component

otx_dir = os.getcwd()

templates = Registry("otx/algorithms/classification").filter(task_type="CLASSIFICATION").templates
templates_ids = [template.model_template_id for template in templates]

multi_class_regression_config = load_regression_configuration(otx_dir, "classification", "supervised", "multi_class")
multi_class_data_args = multi_class_regression_config["data_path"]
multi_class_data_args["train_params"] = ["params"]

class TestRegressionMultiClassClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)
        otx_regression_testing(
            template, tmp_dir_path, otx_dir, multi_class_data_args, multi_class_regression_config["model_criteria"]
        )

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_semisl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_semisl"
        config_semisl = load_regression_configuration(otx_dir, "classification", "semi_supervised", "multi_class")
        args_semisl = config_semisl["data_path"]

        args_semisl["train_params"] = ["params"]
        args_semisl["train_params"].extend(["--algo_backend.train_type", "SEMISUPERVISED"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)

        args_semisl.pop("train_params")
        otx_regression_testing(template, tmp_dir_path, otx_dir, args_semisl, config_semisl["model_criteria"])

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_selfsl(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_selfsl"
        config_selfsl = load_regression_configuration(otx_dir, "classification", "self_supervised", "multi_class")
        args_selfsl = config_selfsl["data_path"]

        args_selfsl["train_params"] = ["params"]
        args_selfsl["train_params"].extend(["--algo_backend.train_type", "SELFSUPERVISED"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args_selfsl)

        args_selfsl.pop("train_params")
        otx_regression_testing(template, tmp_dir_path, otx_dir, args_selfsl, config_selfsl["model_criteria"])

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_supcon(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls/test_supcon"
        config_supcon = load_regression_configuration(otx_dir, "classification", "supervised", "supcon")
        args_supcon = config_supcon["data_path"]

        args_supcon["train_params"] = ["params"]
        args_supcon["train_params"].extend(["--learning_parameters.enable_supcon", "True"])
        otx_train_testing(template, tmp_dir_path, otx_dir, args_supcon)

        args_supcon.pop("train_params")
        otx_regression_testing(template, tmp_dir_path, otx_dir, args_supcon, config_supcon["model_criteria"])

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, multi_class_data_args, threshold=0.0)

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

        nncf_eval_testing(template, tmp_dir_path, otx_dir, multi_class_data_args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_class_cls"
        pot_eval_testing(template, tmp_dir_path, otx_dir, multi_class_data_args)


multi_label_regression_config = load_regression_configuration(otx_dir, "classification", "supervised", "multi_label")
multi_label_data_args = multi_label_regression_config["data_path"]
multi_label_data_args["train_params"] = ["params"]


class TestRegressionMultiLabelClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)
        otx_regression_testing(
            template, tmp_dir_path, otx_dir, multi_label_data_args, multi_label_regression_config["model_criteria"]
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
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, multi_label_data_args, threshold=0.0)

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

        nncf_eval_testing(template, tmp_dir_path, otx_dir, multi_label_data_args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "multi_label_cls"
        pot_eval_testing(template, tmp_dir_path, otx_dir, multi_label_data_args)


h_label_regression_config = load_regression_configuration(otx_dir, "classification", "supervised", "h_label")
h_label_data_args = h_label_regression_config["data_path"]
h_label_data_args["train_params"] = ["params"]


class TestRegressionHierarchicalLabelClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        otx_train_testing(template, tmp_dir_path, otx_dir, h_label_data_args)
        otx_regression_testing(
            template, tmp_dir_path, otx_dir, h_label_data_args, h_label_regression_config["model_criteria"]
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
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, h_label_data_args, threshold=0.0)

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

        nncf_eval_testing(template, tmp_dir_path, otx_dir, h_label_data_args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, h_label_data_args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "h_label_cls"
        pot_eval_testing(template, tmp_dir_path, otx_dir, h_label_data_args)
