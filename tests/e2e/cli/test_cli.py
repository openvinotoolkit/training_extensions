"""Tests for OTX CLI commands"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest

from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    otx_build_auto_config,
    otx_build_backbone_testing,
    otx_find_testing,
    otx_train_auto_config,
)

otx_dir = os.getcwd()


build_backbone_args = [
    ("CLASSIFICATION", "torchvision.mobilenet_v3_large"),
    ("CLASSIFICATION", "mmcls.MMOVBackbone"),
    ("DETECTION", "torchvision.mobilenet_v3_large"),
    ("INSTANCE_SEGMENTATION", "torchvision.mobilenet_v3_large"),
    ("SEGMENTATION", "torchvision.mobilenet_v3_large"),
]
build_backbone_args_ids = [f"{task}_{backbone}" for task, backbone in build_backbone_args]


class TestToolsOTXCLI:
    @e2e_pytest_component
    def test_otx_find(self):
        otx_find_testing()

    @e2e_pytest_component
    @pytest.mark.parametrize("build_backbone_args", build_backbone_args, ids=build_backbone_args_ids)
    def test_otx_backbone_build(self, tmp_dir_path, build_backbone_args):
        tmp_dir_path = tmp_dir_path / build_backbone_args[0] / build_backbone_args[1]
        otx_build_backbone_testing(tmp_dir_path, build_backbone_args)


auto_config_args_with_autosplit = {"--train-data-roots": "tests/assets/imagenet_dataset"}

auto_config_args_with_autosplit_task = {
    "--task": "classification",
    "--train-data-roots": "tests/assets/imagenet_dataset",
}

auto_config_args_without_autosplit = {
    "--train-data-roots": "tests/assets/imagenet_dataset",
    "--val-data-roots": "tests/assets/imagenet_dataset_class_incremental",
}

build_auto_config_args = {
    "classification": {"--train-data-roots": "tests/assets/imagenet_dataset"},
    "classification_with_task": {"--task": "classification", "--train-data-roots": "tests/assets/imagenet_dataset"},
    "detection": {"--train-data-roots": "tests/assets/car_tree_bug"},
    "detection_with_task": {"--task": "detection", "--train-data-roots": "tests/assets/car_tree_bug"},
}


class TestToolsOTXBuildAutoConfig:
    @e2e_pytest_component
    @pytest.mark.parametrize("case", build_auto_config_args.keys())
    def test_otx_build_with_autosplit(self, case, tmp_dir_path):
        otx_dir = os.getcwd()
        tmp_dir_path = tmp_dir_path / "test_build_auto_config" / case
        otx_build_auto_config(root=tmp_dir_path, otx_dir=otx_dir, args=build_auto_config_args[case])


train_auto_config_args = {
    "classification": {"--train-data-roots": "tests/assets/imagenet_dataset"},
    "classification_with_template": {
        "template": "otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml",
        "--train-data-roots": "tests/assets/imagenet_dataset",
    },
    "detection": {"--train-data-roots": "tests/assets/car_tree_bug"},
    "detection_with_template": {
        "template": "otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml",
        "--train-data-roots": "tests/assets/car_tree_bug",
    },
}

train_params = [
    "params",
    "--learning_parameters.num_iters",
    "4",
]


class TestToolsOTXTrainAutoConfig:
    @e2e_pytest_component
    @pytest.mark.parametrize("case", train_auto_config_args.keys())
    def test_otx_train(self, case, tmp_dir_path):
        otx_dir = os.getcwd()
        tmp_dir_path = tmp_dir_path / case
        train_auto_config_args[case]["train_params"] = train_params
        otx_train_auto_config(root=tmp_dir_path, otx_dir=otx_dir, args=train_auto_config_args[case])
