"""Tests for Semantic segmentation with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from otx.cli.utils.tests import (
    get_template_dir,
    nncf_eval_openvino_testing,
    nncf_eval_testing,
    nncf_export_testing,
    nncf_optimize_testing,
    otx_demo_deployment_testing,
    otx_demo_openvino_testing,
    otx_demo_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_hpo_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component

args = {
    "--train-ann-file": "data/segmentation/custom/annotations/training",
    "--train-data-roots": "data/segmentation/custom/images/training",
    "--val-ann-file": "data/segmentation/custom/annotations/training",
    "--val-data-roots": "data/segmentation/custom/images/training",
    "--test-ann-files": "data/segmentation/custom/annotations/training",
    "--test-data-roots": "data/segmentation/custom/images/training",
    "--input": "data/segmentation/custom/images/training",
    "train_params": [
        "params",
        "--learning_parameters.learning_rate_fixed_iters",
        "0",
        "--learning_parameters.learning_rate_warmup_iters",
        "25",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

otx_dir = os.getcwd()

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("otx/algorithms/segmentation/configs", "ocr_lite_hrnet_18_mod2", "template.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("otx/algorithms/segmentation").filter(task_type="SEGMENTATION").templates
    templates_ids = [template.model_template_id for template in templates]


@pytest.fixture(scope="session")
def tmp_dir_path():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


class TestToolsMPASegmentation:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = args.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        otx_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=0.1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94792")
    def test_otx_demo(self, template, tmp_dir_path):
        otx_demo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94792")
    def test_otx_demo_openvino(self, template, tmp_dir_path):
        otx_demo_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94792")
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94792")
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94792")
    def test_otx_demo_deployment(self, template, tmp_dir_path):
        otx_demo_deployment_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94792")
    def test_otx_hpo(self, template, tmp_dir_path):
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, tmp_dir_path, otx_dir, args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="Need to check POT with openvino 2022.2.0")
    def test_pot_optimize(self, template, tmp_dir_path):
        if template.model_template_id.startswith("ClassIncremental_Semantic_Segmentation_Lite-HRNet-"):
            pytest.skip("CVS-82482")
        pot_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="Need to check POT with openvino 2022.2.0")
    def test_pot_eval(self, template, tmp_dir_path):
        if template.model_template_id.startswith("ClassIncremental_Semantic_Segmentation_Lite-HRNet-"):
            pytest.skip("CVS-82482")
        pot_eval_testing(template, tmp_dir_path, otx_dir, args)
