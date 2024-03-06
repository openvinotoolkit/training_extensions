# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from datumaro import Label
from datumaro.components.dataset import Dataset, DatasetItem 
from datumaro.components.media import Image
from datumaro.components.dataset_base import CategoriesInfo
from datumaro.components.annotation import Annotation, AnnotationType, LabelCategories

from otx.core.config import register_configs
from otx.core.data.dataset.base import LabelInfo
from otx.core.data.dataset.classification import HLabelInfo


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
def fxt_hlabel_dataset_subset() -> Dataset:
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=0,
                subset='train',
                media=Image.from_numpy(np.zeros((3, 10, 10))),
                annotations=[
                    Label(
                        label=2,
                        id=0,
                        group=1,
                    ),
                ]
            ),
            DatasetItem(
                id=1,
                subset='train',
                media=Image.from_numpy(np.zeros((3, 10, 10))),
                annotations=[
                    Label(
                        label=4,
                        id=0,
                        group=2,
                    ),
                ]
            )
        ],
        categories=
            {
                AnnotationType.label : LabelCategories(
                    items=[
                        LabelCategories.Category(name='Heart', parent=''),
                        LabelCategories.Category(name='Spade', parent=''),
                        LabelCategories.Category(name='Heart_Queen', parent='Heart'),
                        LabelCategories.Category(name='Heart_King', parent='Heart'),
                        LabelCategories.Category(name='Spade_A', parent='Spade'),
                        LabelCategories.Category(name='Spade_King', parent='Spade'),
                        LabelCategories.Category(name='Black_Joker', parent=''),
                        LabelCategories.Category(name='Red_Joker', parent=''),
                        LabelCategories.Category(name='Extra_Joker', parent=''),
                    ],
                    label_groups=[
                        LabelCategories.LabelGroup(name="Card", labels=["Heart", "Spade"]),
                        LabelCategories.LabelGroup(name="Heart Group", labels=["Heart_Queen", "Heart_King"]),
                        LabelCategories.LabelGroup(name="Spade Group", labels=["Spade_Queen", "Spade_King"]),
                    ]
                )
            },
    ).get_subset("train")


@pytest.fixture()
def fxt_hlabel_multilabel_info() -> HLabelInfo:
    return HLabelInfo(
        label_names=[
            "Heart", "Spade", "Heart_Queen", "Heart_King", "Spade_A", "Spade_King", "Black_Joker", "Red_Joker", "Extra_Joker"
        ],
        label_groups=[["Heart", "Spade"], ["Heart_Queen", "Heart_King"], ["Spade_A", "Spade_King"], ["Black_Joker"], ["Red_Joker"], ["Extra_Joker"]],
        num_multiclass_heads=3,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4)},
        num_single_label_classes=3,
        empty_multiclass_head_indices=[],
        class_to_group_idx={
            "Heart": (0, 0),
            "Spade": (0, 1),
            "Heart_Queen": (1, 0),
            "Heart_King": (1, 1),
            "Spade_A": (2, 0),
            "Spade_King": (2, 1),
            "Black_Joker": (3, 0),
            "Red_Joker": (3, 1),
            "Extra_Joker": (3, 2),
        },
        all_groups=[["Heart", "Spade"], ["Heart_Queen", "Heart_King"], ["Spade_A", "Spade_King"], ["Black_Joker"], ["Red_Joker"], ["Extra_Joker"]],
        label_to_idx={
            "Heart": 0,
            "Spade": 1,
            "Heart_Queen": 2,
            "Heart_King": 3,
            "Spade_A": 4,
            "Spade_King": 5,
            "Black_Joker": 6,
            "Red_Joker": 7,
            "Extra_Joker": 8,
        },
        label_tree_edges=[
            ["Heart_Queen", "Heart"],
            ["Heart_King", "Heart"],
            ["Spade_A", "Spade"],
            ["Spade_King", "Spade"],
        ],
    )
