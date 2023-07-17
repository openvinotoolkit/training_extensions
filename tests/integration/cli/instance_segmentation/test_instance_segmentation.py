"""Tests for Class-Incremental Learning for object detection with OTX CLI"""
# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os

import pytest
import torch

from otx.api.entities.model_template import parse_model_template, ModelCategory, ModelStatus
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
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

args = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": ["params", "--learning_parameters.num_iters", "1", "--learning_parameters.batch_size", "2"],
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
    os.path.join("src/otx/algorithms/detection/configs", "instance_segmentation", "resnet50_maskrcnn", "template.yaml")
)
default_templates = [default_template]
default_templates_ids = [default_template.model_template_id]

templates = Registry("src/otx/algorithms/detection").filter(task_type="INSTANCE_SEGMENTATION").templates
templates_ids = [template.model_template_id for template in templates]

template_experimental = parse_model_template(
    os.path.join(
        "src/otx/algorithms/detection/configs", "instance_segmentation/convnext_maskrcnn", "template_experiment.yaml"
    )
)
templates_inc_convnext = copy.deepcopy(templates)
templates_ids_inc_convnext = copy.deepcopy(templates_ids)
templates_inc_convnext.extend([template_experimental])
templates_ids_inc_convnext.extend([template_experimental.model_template_id])


class TestInstanceSegmentationModelTemplates:
    @e2e_pytest_component
    def test_model_category(self):
        stat = {
            ModelCategory.SPEED: 0,
            ModelCategory.BALANCE: 0,
            ModelCategory.ACCURACY: 0,
            ModelCategory.OTHER: 0,
        }
        for template in templates:
            stat[template.model_category] += 1
        assert stat[ModelCategory.SPEED] == 1
        assert stat[ModelCategory.BALANCE] <= 1
        assert stat[ModelCategory.ACCURACY] == 1

    @e2e_pytest_component
    def test_model_status(self):
        for template in templates:
            if template.model_status == ModelStatus.DEPRECATED:
                assert template.model_category == ModelCategory.OTHER

    @e2e_pytest_component
    def test_default_for_task(self):
        num_default_model = 0
        for template in templates:
            if template.is_default_for_task:
                num_default_model += 1
                assert template.model_category != ModelCategory.OTHER
                assert template.model_status == ModelStatus.ACTIVE
        assert num_default_model == 1


class TestInstanceSegmentationCLI:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_train_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_resume"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args)
        args1["train_params"] = resume_params
        args1[
            "--resume-from"
        ] = f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_export_onnx(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_export_testing(template, tmp_dir_path, half_precision=False, is_onnx=True)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_explain(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_inc_convnext, ids=templates_ids_inc_convnext)
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=1.0)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", default_templates, ids=default_templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_multi_gpu"
        args1 = copy.deepcopy(args)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)
