# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.core.types.label import NullLabelInfo, SegLabelInfo


def test_as_json(fxt_label_info):
    serialized = fxt_label_info.to_json()
    deserialized = fxt_label_info.__class__.from_json(serialized)
    assert fxt_label_info == deserialized


def test_seg_label_info():
    # Automatically insert background label at zero index
    assert SegLabelInfo(["car", "bug", "tree"], []) == SegLabelInfo(["Background", "car", "bug", "tree"], [])
    assert SegLabelInfo.from_num_classes(3) == SegLabelInfo(
        ["Background", "label_0", "label_1"],
        [["Background", "label_0", "label_1"]],
    )
    assert SegLabelInfo.from_num_classes(1) == SegLabelInfo(["Background"], [["Background"]])
    assert SegLabelInfo.from_num_classes(0) == NullLabelInfo()
