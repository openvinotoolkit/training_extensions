# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for unit tests of data entities."""

from __future__ import annotations

import pytest
import torch
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torchvision import tv_tensors

from getitune.data.entity.sample import ClassificationSample


@pytest.fixture
def fxt_torchvision_data_entity() -> ClassificationSample:
    return ClassificationSample(
        image=tv_tensors.Image(torch.randn(3, 10, 10), dtype=torch.float32),
        dm_image_info=DmImageInfo(width=10, height=10),
        subset=Subset.TRAINING,
        label=torch.tensor(0),
    )
