"""Tests for Class-Incremental Learning for object detection with OTX CLI"""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import copy
import os
from pathlib import Path

import pytest
import torch
from pathlib import Path

from otx.api.entities.model_template import parse_model_template
from otx.cli.registry import Registry
from tests.test_suite.e2e_test_system import e2e_pytest_component
from tests.test_suite.run_test_command import (
    get_template_dir,
    nncf_eval_openvino_testing,
    nncf_eval_testing,
    nncf_export_testing,
    nncf_optimize_testing,
    nncf_validate_fq_testing,
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
    otx_resume_testing,
    otx_train_testing,
    ptq_eval_testing,
    ptq_optimize_testing,
    ptq_validate_fq_testing,
)

# Pre-train w/ 'car & tree' class
args0 = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": ["params", "--learning_parameters.num_iters", "5", "--learning_parameters.batch_size", "2"],
}

# Class-Incremental learning w/ 'car', 'tree', 'bug' classes ## TODO: add class incr sample
args = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": ["params", "--learning_parameters.num_iters", "5", "--learning_parameters.batch_size", "2"],
}

# Semi-SL
args_semisl = {
    "--train-data-roots": "tests/assets/car_tree_bug",
    "--val-data-roots": "tests/assets/car_tree_bug",
    "--test-data-roots": "tests/assets/car_tree_bug",
    "--unlabeled-data-roots": "tests/assets/car_tree_bug",
    "--input": "tests/assets/car_tree_bug/images/train",
    "train_params": ["params", "--learning_parameters.num_iters", "5", "--learning_parameters.batch_size", "2"],
}

# Training params for resume, num_iters*2
resume_params = [
    "params",
    "--learning_parameters.num_iters",
    "8",
    "--learning_parameters.batch_size",
    "2",
]

otx_dir = os.getcwd()

iseg_config_root = Path("src/otx/algorithms/detection/configs/instance_segmentation")

MULTI_GPU_UNAVAILABLE = torch.cuda.device_count() <= 1
TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(iseg_config_root / "resnet50_maskrcnn" / "template.yaml")
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("src/otx/algorithms/detection").filter(task_type="INSTANCE_SEGMENTATION").templates
    templates_ids = [template.model_template_id for template in templates]
    # add experimental templates for new inst-seg models. In the future we will update them as main templates
    # but we need to start to test them now.
    templates_with_experimental = copy.deepcopy(templates)
    templates_ids_with_experimental = copy.deepcopy(templates_ids)
    for experimental_template in iseg_config_root.glob("**/*_experimental.yaml"):
        template_experimental = parse_model_template(experimental_template)
        templates_with_experimental.extend([template_experimental])
        templates_ids_with_experimental.extend([template_experimental.model_template_id])


class TestToolsOTXInstanceSegmentation:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_train_testing(template, tmp_dir_path, otx_dir, args0)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args)
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/models/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_resume(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_resume"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args0)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = copy.deepcopy(args0)
        args1["train_params"] = resume_params
        args1[
            "--resume-from"
        ] = f"{template_work_dir}/trained_for_resume_{template.model_template_id}/models/weights.pth"
        otx_resume_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.parametrize("dump_features", [True, False])
    def test_otx_export(self, template, tmp_dir_path, dump_features):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_export_testing(template, tmp_dir_path, dump_features)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_export_fp16(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_export_testing(template, tmp_dir_path, half_precision=True)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.skip(reason="Issue#2290: MaskRCNN shows degraded performance when inferencing in OpenVINO")
    @pytest.mark.parametrize("half_precision", [True, False])
    def test_otx_eval_openvino(self, template, tmp_dir_path, half_precision):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_openvino_testing(template, tmp_dir_path, otx_dir, args, threshold=0.2, half_precision=half_precision)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_explain(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_explain_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.skip(reason="Issue#2290: MaskRCNN shows degraded performance when inferencing in OpenVINO")
    def test_otx_explain_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_explain_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    def test_otx_demo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_demo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.skip(reason="Issue#2290: MaskRCNN shows degraded performance when inferencing in OpenVINO")
    def test_otx_demo_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_demo_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.skip(reason="Issue#2290: MaskRCNN shows degraded performance when inferencing in OpenVINO")
    def test_otx_deploy_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_deploy_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.skip(reason="Issue#2290: MaskRCNN shows degraded performance when inferencing in OpenVINO")
    def test_otx_eval_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_eval_deployment_testing(template, tmp_dir_path, otx_dir, args, threshold=0.0)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates_with_experimental, ids=templates_ids_with_experimental)
    @pytest.mark.skip(reason="Issue#2290: MaskRCNN shows degraded performance when inferencing in OpenVINO")
    def test_otx_demo_deployment(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        otx_demo_deployment_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_hpo(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_hpo"
        otx_hpo_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_export(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_export_testing(template, tmp_dir_path)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_validate_fq(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_validate_fq_testing(template, tmp_dir_path, otx_dir, "instance_segmentation", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(
        reason="Issue#2234 otx eval with nncf optimized model shows different performance with final evaluation when otx optimize"
    )
    def test_nncf_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_testing(template, tmp_dir_path, otx_dir, args, threshold=0.01)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    @pytest.mark.skip(
        reason="Issue#2234 otx eval with nncf optimized model shows different performance with final evaluation when otx optimize"
    )
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")

        nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_optimize(self, template, tmp_dir_path):
        if "MaskRCNN-ConvNeXt" in template.name:
            pytest.skip("CVS-118373 ConvNeXt Compilation Error in PTQ")
        tmp_dir_path = tmp_dir_path / "ins_seg"
        ptq_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_validate_fq(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if "MaskRCNN-ConvNeXt" in template.name:
            pytest.skip("CVS-118373 ConvNeXt Compilation Error in PTQ")
        ptq_validate_fq_testing(template, tmp_dir_path, otx_dir, "instance_segmentation", type(self).__name__)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_ptq_eval(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg"
        if "MaskRCNN-ConvNeXt" in template.name:
            pytest.skip("CVS-118373 ConvNeXt Compilation Error in PTQ")
        ptq_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train(self, template, tmp_dir_path):
        tmp_dir_path = tmp_dir_path / "ins_seg/test_multi_gpu"
        args1 = copy.deepcopy(args)
        args1["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)


class TestToolsOTXSemiSLInstanceSegmentation:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        if not (Path(template.model_template_path).parent / "semisl").is_dir():
            pytest.skip("Semi-SL training type isn't available for this template")
        tmp_dir_path = tmp_dir_path / "ins_seg/test_semisl"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl)
        template_dir = get_template_dir(template, tmp_dir_path)
        assert (Path(template_dir) / "semisl").is_dir()

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        if not (Path(template.model_template_path).parent / "semisl").is_dir():
            pytest.skip("Semi-SL training type isn't available for this template")
        tmp_dir_path = tmp_dir_path / "ins_seg/test_semisl"
        otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.skipif(MULTI_GPU_UNAVAILABLE, reason="The number of gpu is insufficient")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_multi_gpu_train_semisl(self, template, tmp_dir_path):
        if not (Path(template.model_template_path).parent / "semisl").is_dir():
            pytest.skip("Semi-SL training type isn't available for this template")
        tmp_dir_path = tmp_dir_path / "ins_seg/test_multi_gpu_semisl"
        args_semisl_multigpu = copy.deepcopy(args_semisl)
        args_semisl_multigpu["--gpus"] = "0,1"
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semisl_multigpu)
        template_dir = get_template_dir(template, tmp_dir_path)
        assert (Path(template_dir) / "semisl").is_dir()