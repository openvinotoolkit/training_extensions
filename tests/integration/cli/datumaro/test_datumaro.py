"""Tests for Datumaro Integration with OTE CLI"""
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
    nncf_optimize_testing,
    otx_eval_deployment_testing,
    otx_eval_openvino_testing,
    otx_eval_testing,
    otx_train_testing,
    otx_hpo_testing,
    pot_eval_testing,
    pot_optimize_testing,
)
from tests.test_suite.e2e_test_system import e2e_pytest_component

otx_dir = os.getcwd()


@pytest.fixture(scope="session")
def tmp_dir_path():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


""" 
Classification Tests
"""

# Pre-train w/ 'label_0', 'label_1' classes
args_classification = {
    "--train-data-roots": "data/datumaro/imagenet_dataset",
    "--val-data-roots": "data/datumaro/imagenet_dataset",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

# Pre-train w/ 'car', 'tree' classes
args_classification_multilabel = {
    "--train-data-roots": "data/datumaro/datumaro_multilabel",
    "--val-data-roots": "data/datumaro/datumaro_multilabel",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}

args_classification_hierarchical_label = {
    "--train-data-roots": "data/datumaro/datumaro_h-label",
    "--val-data-roots": "data/datumaro/datumaro_h-label",
    "train_params": [
        "params",
        "--learning_parameters.num_iters",
        "2",
        "--learning_parameters.batch_size",
        "4",
    ],
}
classfication_args_list = [
    args_classification,
    args_classification_multilabel,
    args_classification_hierarchical_label
]

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join(
            "otx/algorithms/classification",
            "configs",
            "efficientnet_b0_cls_incr",
            "template.yaml",
        )
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("otx/algorithms/classification").filter(task_type="CLASSIFICATION").templates
    templates_ids = [template.model_template_id for template in templates]


class TestToolsOTXClassificationDatumaro:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_classfication(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_classification)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = args_classification.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)
    
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_classfication_multilabel(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_classification_multilabel)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = args_classification_multilabel.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train_classfication_hierarchical_label(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_classification_hierarchical_label)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = args_classification_hierarchical_label.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        for args in classfication_args_list:
            otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")
        for args in classfication_args_list:
            nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)
    
    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")
        for args in classfication_args_list:
            nncf_eval_testing(template, tmp_dir_path, otx_dir, args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")
        for args in classfication_args_list:
            nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        for args in classfication_args_list:
            pot_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        for args in classfication_args_list:
            pot_eval_testing(template, tmp_dir_path, otx_dir, args)
    

""" 
Detection Tests
"""

args_detection = {
    "--train-data-roots": "data/datumaro/coco_dataset/coco_detection",
    "train_params": ["params", "--learning_parameters.num_iters", "4", "--learning_parameters.batch_size", "4"],
}

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


class TestToolsOTXDetectionDatumaro:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_detection)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = args_detection.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_eval(self, template, tmp_dir_path):
        for args in classfication_args_list:
            otx_eval_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_optimize(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")
        for args in classfication_args_list:
            nncf_optimize_testing(template, tmp_dir_path, otx_dir, args)
    
    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")
        for args in classfication_args_list:
            nncf_eval_testing(template, tmp_dir_path, otx_dir, args, threshold=0.001)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_nncf_eval_openvino(self, template, tmp_dir_path):
        if template.entrypoints.nncf is None:
            pytest.skip("nncf entrypoint is none")
        for args in classfication_args_list:
            nncf_eval_openvino_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_optimize(self, template, tmp_dir_path):
        for args in classfication_args_list:
            pot_optimize_testing(template, tmp_dir_path, otx_dir, args)

    @e2e_pytest_component
    @pytest.mark.skipif(TT_STABILITY_TESTS, reason="This is TT_STABILITY_TESTS")
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_pot_eval(self, template, tmp_dir_path):
        for args in classfication_args_list:
            pot_eval_testing(template, tmp_dir_path, otx_dir, args)

""" 
Instance Segmentation Tests
"""

args_instance_segmentation = {
    "--train-data-roots": "data/datumaro/coco_dataset/coco_instance_segmentation",
    "train_params": ["params", "--learning_parameters.num_iters", "4", "--learning_parameters.batch_size", "2"],
}

TT_STABILITY_TESTS = os.environ.get("TT_STABILITY_TESTS", False)
if TT_STABILITY_TESTS:
    default_template = parse_model_template(
        os.path.join("otx/algorithms/detection/configs", "instance_segmentation", "resnet50_maskrcnn", "template.yaml")
    )
    templates = [default_template] * 100
    templates_ids = [template.model_template_id + f"-{i+1}" for i, template in enumerate(templates)]
else:
    templates = Registry("otx/algorithms/detection").filter(task_type="INSTANCE_SEGMENTATION").templates
    templates_ids = [template.model_template_id for template in templates]


class TestToolsOTXInstanceSegmentationDatumaro:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_instance_segmentation)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = args_instance_segmentation.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)


"""
Semantic Segmentation Tests
"""

args_semantic_segmentation = {
    "--train-data-roots": "data/datumaro/common_semantic_segmentation_dataset/dataset",
    "--val-data-roots": "data/datumaro/common_semantic_segmentation_dataset/dataset",
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


class TestToolsOTXSemanticSegmentation:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_semantic_segmentation)
        template_work_dir = get_template_dir(template, tmp_dir_path)
        args1 = args_semantic_segmentation.copy()
        args1["--load-weights"] = f"{template_work_dir}/trained_{template.model_template_id}/weights.pth"
        otx_train_testing(template, tmp_dir_path, otx_dir, args1)


"""
Anomaly Classification Tests
"""

args_anomaly_tasks = {
    "--train-data-roots": "data/datumaro/mvtec/train",
    "--val-data-roots": "data/datumaro/mvtec/test",
    "--test-data-roots": "data/datumaro/mvtec/test",
    "train_params": [],
}

templates = Registry("otx/algorithms").filter(task_type="ANOMALY_CLASSIFICATION").templates
templates_ids = [template.model_template_id for template in templates]


class TestToolsOTXAnomalyClassification:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_anomaly_tasks)


"""
Anomaly Detection Tests
"""

templates = Registry("otx/algorithms").filter(task_type="ANOMALY_DETECTION").templates
templates_ids = [template.model_template_id for template in templates]


class TestToolsOTXAnomalyDetection:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_anomaly_tasks)


"""
Anomaly Segmentation Tests
"""

templates = Registry("otx/algorithms").filter(task_type="ANOMALY_SEGMENTATION").templates
templates_ids = [template.model_template_id for template in templates]


class TestToolsOTXAnomalySegmentation:
    @e2e_pytest_component
    @pytest.mark.parametrize("template", templates, ids=templates_ids)
    def test_otx_train(self, template, tmp_dir_path):
        otx_train_testing(template, tmp_dir_path, otx_dir, args_anomaly_tasks)
