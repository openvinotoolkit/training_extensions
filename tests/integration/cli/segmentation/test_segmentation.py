"""Tests for Semantic segmentation with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template
from otx.cli.utils.tests import (
    get_template_dir,
    nncf_optimize_testing,
    otx_deploy_openvino_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_export_testing,
    otx_hpo_testing,
    otx_resume_testing,
    otx_train_testing,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component

args = {
    "--train-data-roots": "data/common_semantic_segmentation_dataset/train",
    "--val-data-roots": "data/common_semantic_segmentation_dataset/val",
    "--test-data-roots": "data/common_semantic_segmentation_dataset/val",
    "--input": "data/common_semantic_segmentation_dataset/train/images",
    "train_params": [
        "params",
        "--learning_parameters.learning_rate_fixed_iters",
        "0",
        "--learning_parameters.learning_rate_warmup_iters",
        "1",
        "--learning_parameters.num_iters",
        "1",
        "--learning_parameters.batch_size",
        "4",
    ],
}

args_semisl = {
    "--train-data-roots": "data/common_semantic_segmentation_dataset/train",
    "--val-data-roots": "data/common_semantic_segmentation_dataset/val",
    "--test-data-roots": "data/common_semantic_segmentation_dataset/val",
    "--unlabeled-data-roots": "data/common_semantic_segmentation_dataset/train",
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

args_selfsl = {
    "--train-data-roots": "data/common_semantic_segmentation_dataset/train",
    "--input": "data/segmentation/custom/images/training",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
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
    os.path.join("otx/algorithms/segmentation/configs", "ocr_lite_hrnet_18_mod2", "template.yaml")
)
templates = [default_template]
templates_ids = [default_template.model_template_id]


class TestSegmentationCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train_semi(self, template, tmp_dir_path):
        args1 = copy.deepcopy(args_semisl)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train_self(self, template, tmp_dir_path):
        args1 = copy.deepcopy(args_selfsl)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)
