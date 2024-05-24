# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from otx.core.types.label import NullLabelInfo, SegLabelInfo


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
