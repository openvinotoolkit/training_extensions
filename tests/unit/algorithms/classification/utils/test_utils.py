# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
from pathlib import Path

import pytest

from otx.algorithms.classification.utils.cls_utils import (
    get_cls_deploy_config,
    get_cls_inferencer_configuration,
    get_cls_model_api_configuration,
    get_multihead_class_info,
)
from otx.algorithms.classification.utils.convert_coco_to_multilabel import (
    coco_to_datumaro_multilabel,
    multilabel_ann_format,
)
from tests.test_suite.e2e_test_system import e2e_pytest_unit
from tests.unit.algorithms.classification.test_helper import (
    generate_cls_dataset,
    generate_label_schema,
)


@pytest.fixture
def default_hierarchical_data():
    hierarchical_dataset = generate_cls_dataset(hierarchical=True)
    label_schema = generate_label_schema(hierarchical_dataset.get_labels(), multilabel=False, hierarchical=True)
    return hierarchical_dataset, label_schema


@e2e_pytest_unit
def test_coco_conversion(tmp_dir_path):
    path_to_coco_example_ann_file = (
        Path("tests/assets/car_tree_bug/annotations/instances_train.json").absolute().as_posix()
    )
    path_to_coco_example_image_dir = Path("tests/assets/car_tree_bug/images/train").absolute().as_posix()
    output_path = Path(tmp_dir_path) / "annotations.json"
    coco_to_datumaro_multilabel(
        path_to_coco_example_ann_file, path_to_coco_example_image_dir, output_path, test_mode=True
    )
    assert Path.exists(output_path)
    with open(output_path, "r") as f:
        coco_ann = json.load(f)
        for key in multilabel_ann_format:
            assert key in coco_ann.keys()
        assert len(coco_ann["items"]) > 0


@e2e_pytest_unit
def test_get_multihead_class_info(default_hierarchical_data):
    hierarchical_dataset, label_schema = default_hierarchical_data
    class_info = get_multihead_class_info(label_schema)
    assert (
        len(class_info["label_to_idx"])
        == len(class_info["class_to_group_idx"])
        == len(hierarchical_dataset.get_labels())
    )


@e2e_pytest_unit
def test_get_cls_inferencer_configuration(default_hierarchical_data) -> None:
    _, label_schema = default_hierarchical_data
    config = get_cls_inferencer_configuration(label_schema)

    assert config["hierarchical"]
    assert not config["multilabel"]
    assert "multihead_class_info" in config


@e2e_pytest_unit
def test_get_cls_deploy_config(default_hierarchical_data) -> None:
    _, label_schema = default_hierarchical_data
    inf_conf = {"test": "test"}
    config = get_cls_deploy_config(label_schema, inf_conf)

    assert config["type_of_model"] == "otx_classification"
    assert config["converter_type"] == "CLASSIFICATION"
    assert "labels" in config["model_parameters"]
    for k in inf_conf:
        assert k in config["model_parameters"]


@e2e_pytest_unit
def test_get_cls_model_api_configuration(default_hierarchical_data):
    _, label_schema = default_hierarchical_data
    config = get_cls_inferencer_configuration(label_schema)

    model_api_cfg = get_cls_model_api_configuration(label_schema, config)

    assert len(model_api_cfg) > 0
    assert model_api_cfg[("model_info", "confidence_threshold")] == str(config["confidence_threshold"])
    assert ("model_info", "hierarchical_config") in model_api_cfg
