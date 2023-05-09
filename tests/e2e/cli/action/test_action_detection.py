"""Tests for Action Detection Task with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_train_testing,
    pot_eval_testing,
    pot_optimize_testing,
)

# Finetuning arguments
# TODO: Need to change sample dataset
args = {
    "--train-data-roots": "tests/assets/cvat_dataset/action_detection/train",
    "--val-data-roots": "tests/assets/cvat_dataset/action_detection/train",
    "--test-data-roots": "tests/assets/cvat_dataset/action_detection/train",
    "train_params": ["params", "--learning_parameters.num_iters", "2", "--learning_parameters.batch_size", "4"],
}

otx_dir = os.getcwd()

MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1
TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("otx/algorithms/action/configs", "detection", "x3d_fast_rcnn", "template.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("otx/algorithms/action").filter(task_type="ACTION_DETECTION").templates
    templates_ids = [template.model_template_id for template in templates]


class TestToolsOTXActionDetection:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_det"
        otx_train_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_det"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_det"
        otx_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skip(reason="Issue#2059: OpenVINO exported model shows 0.0 AP50 in torch1.13")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_det"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=0.05)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skip(reason="Issue#2059: OpenVINO exported model shows 0.0 AP50 in torch1.13")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_det"
        pot_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skip(reason="Issue#2059: OpenVINO exported model shows 0.0 AP50 in torch1.13")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_det"
        pot_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "action_det/test_multi_gpu"
        args1 = copy.deepcopy(args)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)
