# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from omegaconf import DictConfig
from otx.core.utils.build import get_default_num_async_infer_requests

SKIP_MMLAB_TEST = False
try:
    from mmpretrain.registry import MODELS  # noqa: F401
except ImportError:
    SKIP_MMLAB_TEST = True


@pytest.mark.skipif(SKIP_MMLAB_TEST, reason="MMLab is not installed")
@pytest.fixture()
def fxt_mm_config() -> DictConfig:
    return DictConfig(
        {
            "backbone": {
                "arch": "b0",
                "type": "EfficientNet",
            },
            "head": {
                "in_channels": 1280,
                "loss": {
                    "loss_weight": 1.0,
                    "type": "CrossEntropyLoss",
                },
                "num_classes": 1000,
                "topk": (1, 5),
                "type": "LinearClsHead",
            },
            "neck": {
                "type": "GlobalAveragePooling",
            },
            "data_preprocessor": {
                "mean": [123.678, 116.28, 103.53],
                "std": [58.395, 57.12, 57.375],
                "to_rgb": False,
                "type": "ClsDataPreprocessor",
            },
            "type": "ImageClassifier",
        },
    )


def test_get_default_num_async_infer_requests() -> None:
    # Test the get_default_num_async_infer_requests function.

    # Mock os.cpu_count() to return a specific value
    original_cpu_count = os.cpu_count
    os.cpu_count = lambda: 4

    # Call the function and check the return value
    assert get_default_num_async_infer_requests() == 2

    # Restore the original os.cpu_count() function
    os.cpu_count = original_cpu_count

    # Check the warning message
    with pytest.warns(UserWarning, match="Set the default number of OpenVINO inference requests"):
        get_default_num_async_infer_requests()
