# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for unit tests of data entities."""

import numpy as np
import pytest
import torch
from datumaro import Polygon
from otx.core.data.entity.base import ImageInfo, OTXDataEntity, Points
from otx.core.data.entity.visual_prompting import VisualPromptingDataEntity
from torch import LongTensor
from torchvision import tv_tensors


@pytest.fixture()
def fxt_numpy_data_entity() -> OTXDataEntity:
    return OTXDataEntity(
        np.ndarray((10, 10, 3)),
        ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
    )


@pytest.fixture()
def fxt_torchvision_data_entity() -> OTXDataEntity:
    return OTXDataEntity(
        tv_tensors.Image(torch.randn(3, 10, 10)),
        ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
    )


@pytest.fixture()
def fxt_visual_prompting_data_entity() -> VisualPromptingDataEntity:
    return VisualPromptingDataEntity(
        image=tv_tensors.Image(torch.randn(3, 10, 10)),
        img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
        masks=tv_tensors.Mask(torch.ones(10, 10)),
        labels=[LongTensor([1]), LongTensor([2])],
        polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
        bboxes=tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 5, 5]), format="xyxy", canvas_size=(10, 10)),
        points=Points(data=torch.Tensor([7, 7]), canvas_size=(10, 10)),
    )
