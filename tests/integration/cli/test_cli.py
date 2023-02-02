"""Tests for OTX CLI commands"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import os

import pytest

from otx.cli.tools.build import SUPPORTED_TASKS
from otx.cli.utils.tests import (
    otx_build_auto_config,
    otx_build_backbone_testing,
    otx_build_task_testing,
    otx_find_testing,
    otx_train_auto_config,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component

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
    @pytest.mark.parametrize("task", SUPPORTED_TASKS, ids=SUPPORTED_TASKS)
    def test_otx_workspace_build(self, tmp_dir_path, task):
        otx_build_task_testing(tmp_dir_path, task)

    @e2e_pytest_component
    @pytest.mark.parametrize("build_backbone_args", build_backbone_args, ids=build_backbone_args_ids)
    def test_otx_backbone_build(self, tmp_dir_path, build_backbone_args):
        otx_build_backbone_testing(tmp_dir_path, build_backbone_args)

######################### Auto configuration ################################
build_auto_config_args = {
    "classification":{
        "--train-data-roots": "data/imagenet_dataset"
    },
    "classification_with_task":{
        "--task": "classification", 
        "--train-data-roots": "data/imagenet_dataset"
    },
    "detection":{
        "--train-data-roots": "data/coco_dataset/coco_detection"
    },
    "detection_with_task":{
        "--task": "detection", 
        "--train-data-roots": "data/coco_dataset/coco_detection"
    },
    "segmentation":{
        "--train-data-roots": "data/cityscapes_dataset/train_dataset"
    },
    "segmentation_with_task":{
        "--task": "segmentation", 
        "--train-data-roots": "data/cityscapes_dataset/train_dataset"
    }
}

class TestToolsOTXBuildAutoConfig:
    @e2e_pytest_component
    @pytest.mark.parametrize("case", build_auto_config_args.keys())
    def test_otx_build(self, case, tmp_dir_path):
        otx_dir = os.getcwd()
        otx_build_auto_config(
            root=tmp_dir_path, 
            otx_dir=otx_dir, 
            args=build_auto_config_args[case]
        )

train_auto_config_args = {
    "classification":{
        "--train-data-roots": "data/imagenet_dataset"
    },
    "classification_with_template":{
        "--template": "otx/algorithms/classification/configs/mobilenet_v3_large_1_cls_incr/template.yaml", 
        "--train-data-roots": "data/imagenet_dataset"
    },
    "detection":{
        "--train-data-roots": "data/coco_dataset/coco_detection"
    },
    "detection_with_template":{
        "--template": "otx/algorithms/detection/configs/detection/mobilenetv2_ssd/template.yaml", 
        "--train-data-roots": "data/coco_dataset/coco_detection"
    },
    # TODO: Segmentation is not fully supported for auto-split()
    #"segmentation":{
    #    "--train-data-roots": "data/cityscapes_dataset/train_dataset"
    #},
    #"segmentation_with_template":{
    #    "--template": "otx/algorithms/segmentation/configs/ocr_lite_hrnet_18_mod2/template.yaml", 
    #    "--train-data-roots": "data/cityscapes_dataset/train_dataset"
    #}
}
train_params = [
    "params",
    "--learning_parameters.num_iters",
    "1",
]

class TestToolsOTXTrainAutoConfig:
    @e2e_pytest_component
    @pytest.mark.parametrize("case", train_auto_config_args.keys())
    def test_otx_train(self, case, tmp_dir_path):
        otx_dir = os.getcwd()
        train_auto_config_args[case]["train_params"] = train_params
        otx_train_auto_config(
            root=tmp_dir_path, 
            otx_dir=otx_dir, 
            args=train_auto_config_args[case]
        )