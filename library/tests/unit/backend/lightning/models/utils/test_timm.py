# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.lightning.models.classification.utils.timm import (
    get_preprocessing_params,
)


@pytest.mark.parametrize(
    ("backbone_name", "expected_input_params"),
    [
        (
            "tf_efficientnetv2_s.in21k",
            DataInputParams(input_size=(300, 300), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ),
        ("vit_base_patch16_224", DataInputParams(input_size=(224, 224), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))),
        (
            "swin_tiny_patch4_window7_224",
            DataInputParams(input_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ),
    ],
)
def test_get_preprocessing_params_returns_expected_values(backbone_name: str, expected_input_params: DataInputParams):
    assert get_preprocessing_params(backbone_name) == expected_input_params
