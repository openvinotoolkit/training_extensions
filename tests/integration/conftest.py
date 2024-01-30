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
