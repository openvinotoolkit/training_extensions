# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.core.config import register_configs
from otx.core.data.dataset.base import LabelInfo
from otx.core.data.entity.classification import HLabelInfo


@pytest.fixture(scope="session", autouse=True)
def fxt_register_configs() -> None:
    register_configs()


@pytest.fixture(scope="session", autouse=True)
def fxt_multiclass_labelinfo() -> LabelInfo:
    label_names = ["class1", "class2", "class3"]
    return LabelInfo(
        label_names=label_names,
        label_groups=[
            label_names,
            ["class2", "class3"],
        ],
    )


@pytest.fixture(scope="session", autouse=True)
def fxt_multilabel_labelinfo() -> LabelInfo:
    label_names = ["class1", "class2", "class3"]
    return LabelInfo(
        label_names=label_names,
        label_groups=[
            [label_names[0]],
            [label_names[1]],
            [label_names[2]],
        ],
    )


@pytest.fixture()
def fxt_hlabel_multilabel_info() -> HLabelInfo:
    return HLabelInfo(
        num_multiclass_heads=3,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4), "2": (4, 6)},
        num_single_label_classes=3,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "0": (0, 0),
            "1": (0, 1),
            "2": (1, 0),
            "3": (1, 1),
            "4": (2, 0),
            "5": (2, 1),
            "6": (3, 0),
            "7": (3, 1),
            "8": (3, 2),
        },
        all_groups=[["0", "1"], ["2", "3"], ["4", "5"], ["6"], ["7"], ["8"]],
        label_to_idx={
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
        },
        label_tree_edges=[
            ["0", "0"],
            ["1", "0"],
            ["2", "1"],
            ["3", "1"],
            ["4", "2"],
            ["5", "2"],
        ],
    )
