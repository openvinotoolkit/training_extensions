"""Tests for Classification with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from otx.cli.utils.tests import (
    get_template_dir,
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_explain_openvino_testing,
    otx_explain_testing,
    otx_export_testing,
    otx_hpo_testing,
    otx_resume_testing,
    otx_train_testing,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component

# Pre-train w/ 'label_0', 'label_1', 'label_2' classes
args = {
    "--train-data-roots": "data/imagenet_dataset_class_incremental",
    "--val-data-roots": "data/imagenet_dataset_class_incremental",
    "--test-data-roots": "data/imagenet_dataset_class_incremental",
    "--input": "data/imagenet_dataset/label_0",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
    ],
}

args_semisl = {
    "--train-data-roots": "data/imagenet_dataset_class_incremental",
    "--val-data-roots": "data/imagenet_dataset_class_incremental",
    "--test-data-roots": "data/imagenet_dataset_class_incremental",
    "--unlabeled-data-roots": "data/imagenet_dataset_class_incremental",
    "--input": "data/imagenet_dataset/label_0",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "5",
        "--learning_parameters.batch_size",
        "4",
        "--algo_backend.train_type",
        "SEMISUPERVISED",
    ],
}

# Warmstart using data w/ 'intel', 'openvino', 'opencv' classes
args_selfsl = {
    "--train-data-roots": "data/imagenet_dataset",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "5",
        "--learning_parameters.batch_size",
        "4",
        "--algo_backend.train_type",
        "SELFSUPERVISED",
    ],
}

# Training params for resume, num_iters*2
resume_params = [
    "params",
    "--learning_parameters.num_iters",
    "2",
    "--learning_parameters.batch_size",
    "4",
]

otx_dir = os.getcwd()


MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1
default_template = parse_model_template(
    os.path.join(
        "otx/algorithms/classification",
        "configs",
        "efficientnet_b0_cls_incr",
        "template.yaml",
    )
)
default_templates = [default_template]
default_templates_ids = [default_template.model_template_id]

templates = Registry("otx/algorithms/classification").filter(task_type="CLASSIFICATION").templates
templates_ids = [template.model_template_id for template in templates]


class TestMultiClassClassificationCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train_semi(self, template, tmp_dir_path):
        args1 = copy.deepcopy(args_semisl)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train_self(self, template, tmp_dir_path):
        args1 = copy.deepcopy(args_selfsl)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)
