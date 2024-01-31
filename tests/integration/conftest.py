# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import pytest
from mmengine.config import Config as MMConfig


@pytest.fixture(scope="session")
def fxt_asset_dir() -> Path:
    return Path(__file__).parent.parent / "assets"


@pytest.fixture(scope="session")
def fxt_rtmdet_tiny_config(fxt_asset_dir: Path) -> MMConfig:
    config_path = fxt_asset_dir / "mmdet_configs" / "rtmdet_tiny_8xb32-300e_coco.py"

    return MMConfig.fromfile(config_path)


# [TODO]: This is a temporary approach.
@pytest.fixture()
def fxt_target_dataset_per_task() -> dict:
    return {
        "multi_class_cls": "tests/assets/classification_dataset",
        "multi_label_cls": "tests/assets/multilabel_classification",
        "h_label_cls": "tests/assets/hlabel_classification",
        "detection": "tests/assets/car_tree_bug",
        "instance_segmentation": "tests/assets/car_tree_bug",
        "semantic_segmentation": "tests/assets/common_semantic_segmentation_dataset/supervised",
        "action_classification": "tests/assets/action_classification_dataset/",
        "action_detection": "tests/assets/action_detection_dataset/",
        "visual_prompting": "tests/assets/car_tree_bug",
        "zero_shot_visual_prompting": "tests/assets/car_tree_bug",
    }


@pytest.fixture()
def fxt_cli_override_command_per_task() -> dict:
    return {
        "multi_class_cls": ["--model.num_classes", "2"],
        "multi_label_cls": ["--model.num_classes", "2"],
        "h_label_cls": [
            "--model.num_classes",
            "7",
            "--model.num_multiclass_heads",
            "2",
            "--model.num_multilabel_classes",
            "3",
        ],
        "detection": ["--model.num_classes", "3"],
        "instance_segmentation": ["--model.num_classes", "3"],
        "semantic_segmentation": ["--model.num_classes", "2"],
        "action_classification": ["--model.num_classes", "2"],
        "action_detection": [
            "--model.num_classes",
            "5",
            "--model.topk",
            "3",
        ],
        "visual_prompting": [],
    }
