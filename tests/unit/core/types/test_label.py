# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datumaro import LabelCategories
from datumaro.components.annotation import GroupType
from otx.core.types.label import HLabelInfo, NullLabelInfo, SegLabelInfo


def test_as_json(fxt_label_info):
    serialized = fxt_label_info.to_json()
    deserialized = fxt_label_info.__class__.from_json(serialized)
    assert fxt_label_info == deserialized


def test_seg_label_info():
    # Automatically insert background label at zero index
    assert SegLabelInfo.from_num_classes(3) == SegLabelInfo(
        ["label_0", "label_1", "label_2"],
        [["label_0", "label_1", "label_2"]],
    )
    assert SegLabelInfo.from_num_classes(1) == SegLabelInfo(["background", "label_0"], [["background", "label_0"]])
    assert SegLabelInfo.from_num_classes(0) == NullLabelInfo()


# Unit test
def test_hlabel_info():
    labels = [
        LabelCategories.Category(name="car", parent="vehicle"),
        LabelCategories.Category(name="truck", parent="vehicle"),
        LabelCategories.Category(name="plush toy", parent="plush toy"),
        LabelCategories.Category(name="No class"),
    ]
    label_groups = [
        LabelCategories.LabelGroup(
            name="Detection labels___vehicle",
            labels=["car", "truck"],
            group_type=GroupType.EXCLUSIVE,
        ),
        LabelCategories.LabelGroup(
            name="Detection labels___plush toy",
            labels=["plush toy"],
            group_type=GroupType.EXCLUSIVE,
        ),
        LabelCategories.LabelGroup(name="No class", labels=["No class"], group_type=GroupType.RESTRICTED),
    ]
    dm_label_categories = LabelCategories(items=labels, label_groups=label_groups)

    hlabel_info = HLabelInfo.from_dm_label_groups(dm_label_categories)

    # Check if class_to_group_idx and label_to_idx have the same keys
    assert list(hlabel_info.class_to_group_idx.keys()) == list(
        hlabel_info.label_to_idx.keys(),
    ), "class_to_group_idx and label_to_idx keys do not match"
