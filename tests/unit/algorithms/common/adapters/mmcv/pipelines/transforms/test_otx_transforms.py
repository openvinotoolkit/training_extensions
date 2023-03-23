"""Unit Tests for the OTX Dataset Pipelines OTX Transforms."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import pytest
import torch
from PIL import Image
from torchvision.transforms import functional as F

from otx.algorithms.classification.adapters.mmcls.datasets.pipelines.transforms.otx_transforms import (
    PILToTensor,
    RandomRotate,
    TensorNormalize,
)


@pytest.fixture
def data() -> dict[str, list[str] | Image]:
    # a sample result dictionary to use in tests
    return {
        "img_fields": ["img"],
        "img": Image.new(mode="RGB", size=(224, 224), color="red"),
    }


def test_PILToTensor(data: dict[str, list[str] | Image]) -> None:
    """Test PILToTensor transform."""
    transform = PILToTensor()
    result = transform(data.copy())  # copy to avoid modifying the original data

    assert result["PILToTensor"] is True
    assert result["img_fields"] == ["img"]
    assert torch.equal(result["img"], F.to_tensor(data["img"]))


def test_TensorNormalize(data: dict[str, list[str] | Image]) -> None:
    """Test TensorNormalize transform."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    data = PILToTensor()(data)  # convert to tensor
    transform = TensorNormalize(mean=mean, std=std)
    result = transform(data.copy())  # copy to avoid modifying the original data

    assert result["TensorNormalize"] is True
    assert result["img_fields"] == ["img"]
    assert torch.equal(result["img"], F.normalize(data["img"], mean, std))


def test_RandomRotate(data: dict[str, list[str] | Image]) -> None:
    """Test RandomRotate transform."""
    transform = RandomRotate(p=1.0, angle=(-10, 10))
    result = transform(data.copy())  # copy to avoid modifying the original data.
    assert result["img"] != data["img"]  # image should be rotated
