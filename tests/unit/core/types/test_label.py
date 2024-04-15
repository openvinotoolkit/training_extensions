# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


def test_as_json(fxt_label_info):
    serialized = fxt_label_info.to_json()
    deserialized = fxt_label_info.__class__.from_json(serialized)
    assert fxt_label_info == deserialized
