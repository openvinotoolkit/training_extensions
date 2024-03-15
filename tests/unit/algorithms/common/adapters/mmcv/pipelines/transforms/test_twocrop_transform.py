"""Unit Tests for the OTX Dataset Pipelines Transforms - Two Crop."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import Compose
from mmcv.utils import build_from_cfg

from otx.algorithms.classification.adapters.mmcls.datasets.pipelines.transforms.twocrop_transform import (
    TwoCropTransform,
)


def test_TwoCropTransform() -> None:
    """Test the TwoCropTransform instance."""
    # Data to be transformed
    data = {}
    data["img"] = np.ones((224, 224, 3), dtype=np.uint8)
    data["gt_label"] = 0

    # Pipeline to be used for transformation
    pipeline = [
        dict(type="Resize", size=(256, 256)),
        dict(type="RandomCrop", size=(224, 224)),
        dict(
            type="Normalize",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
        ),
        dict(type="ToTensor", keys=["img"]),
    ]

    # Create TwoCropTransform instance
    transform = TwoCropTransform(pipeline)
    transform_pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
    transform.pipeline1 = transform_pipeline
    transform.pipeline2 = transform_pipeline

    # Test the TwoCropTransform instance
    transformed_data = transform(data)
    assert isinstance(transformed_data, dict)
    assert transformed_data["img"].shape == (2, 224, 224, 3)
