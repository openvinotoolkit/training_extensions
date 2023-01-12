"""Tests for MPA Class-Incremental Learning for object detection with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest
import torch

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
    otx_explain_openvino_testing,
    otx_explain_testing,
    otx_export_testing,
    otx_hpo_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component

# Pre-train w/ 'person' class
args0 = {
    "--train-ann-file": "data/airport/annotation_person_train.json",
    "--train-data-roots": "data/airport/train",
    "--val-ann-file": "data/airport/annotation_person_train.json",
    "--val-data-roots": "data/airport/train",
    "--test-ann-files": "data/airport/annotation_person_train.json",
    "--test-data-roots": "data/airport/train",
    "--input": "data/airport/train",
    "train_params": ["params", "--learning_parameters.num_iters", "4", "--learning_parameters.batch_size", "4"],
}

# Class-Incremental learning w/ 'vehicle', 'person', 'non-vehicle' classes
args = {
    "--train-ann-file": "data/airport/annotation_example_train.json",
    "--train-data-roots": "data/airport/train",
    "--val-ann-file": "data/airport/annotation_example_train.json",
    "--val-data-roots": "data/airport/train",
    "--test-ann-files": "data/airport/annotation_example_train.json",
    "--test-data-roots": "data/airport/train",
    "--input": "data/airport/train",
    "train_params": ["params", "--learning_parameters.num_iters", "2", "--learning_parameters.batch_size", "4"],
}

args_semisl = {
    "--train-ann-file": "data/airport/annotation_example_train.json",
    "--train-data-roots": "data/airport/train",
    "--val-ann-file": "data/airport/annotation_example_train.json",
    "--val-data-roots": "data/airport/train",
    "--test-ann-files": "data/airport/annotation_example_train.json",
    "--test-data-roots": "data/airport/train",
    "--unlabeled-data-roots": "data/airport/train",
    "--input": "data/airport/train",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
        "--algo_backend.train_type",
        "SEMISUPERVISED",
    ],
}

otx_dir = os.getcwd()

MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1
TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("otx/algorithms/detection/configs", "detection", "mobilenetv2_atss", "template.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("otx/algorithms/detection").filter(task_type="DETECTION").templates
    templates_ids = [template.model_template_id for template in templates]


class TestToolsMPADetection:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args0)
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
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=0.2)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain(self, template, tmp_dir_path):
        otx_explain_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo(self, template, tmp_dir_path):
        otx_demo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo_openvino(self, template, tmp_dir_path):
        otx_demo_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_demo_deployment(self, template, tmp_dir_path):
        otx_demo_deployment_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94790")
    def test_nncf_optimize(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94790")
    def test_nncf_export(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94790")
    def test_nncf_eval(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, tmp_dir_path, otx_dir, args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(reason="CVS-94790")
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        pot_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        pot_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        args1 = args.copy()
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)


class TestToolsMPASemiSLDetection:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)
