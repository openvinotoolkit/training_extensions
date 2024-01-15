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
        np.ndarray((10, 10, 3)),
        ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10), attributes={}),
    )


@pytest.fixture()
def fxt_torchvision_data_entity() -> OTXDataEntity:
    return OTXDataEntity(
        tv_tensors.Image(torch.randn(3, 10, 10)),
        ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10), attributes={}),
    )
