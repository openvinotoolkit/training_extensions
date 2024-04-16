# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from otx.core.types.label import LabelInfo


@pytest.fixture(
    params=[
        "fxt_multiclass_labelinfo",
        "fxt_hlabel_multilabel_info",
        "fxt_null_label_info",
        "fxt_seg_label_info",
    ],
)
def fxt_label_info(request: pytest.FixtureRequest) -> LabelInfo:
    return request.getfixturevalue(request.param)
