# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for unit tests of data entities."""

import numpy as np
import pytest
import torch
from otx.core.data.entity.base import ImageInfo, OTXDataEntity
from torchvision import tv_tensors


@pytest.fixture()
def fxt_numpy_data_entity() -> OTXDataEntity:
    return OTXDataEntity(
        np.ndarray((224, 224, 3)),
        ImageInfo(0, (224, 224, 3), (224, 224, 3), (0, 0, 0), (1.0, 1.0)),
    )


@pytest.fixture()
def fxt_torchvision_data_entity() -> OTXDataEntity:
    return OTXDataEntity(
        tv_tensors.Image(torch.randn(3, 224, 224)),
        ImageInfo(0, (224, 224, 3), (224, 224, 3), (0, 0, 0), (1.0, 1.0)),
    )
