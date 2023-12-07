# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of base data entity."""

import numpy as np
import torch
from otx.core.data.entity.base import ImageInfo, ImageType, OTXBatchDataEntity, OTXDataEntity
from torchvision import tv_tensors


class TestOTXDataEntity:
    def test_image_type(self) -> None:
        data_entity = OTXDataEntity(
            np.ndarray((224, 224, 3)),
            ImageInfo(0, (224, 224, 3), (224, 224, 3), (0, 0, 0), (1.0, 1.0)),
        )
        assert data_entity.image_type == ImageType.NUMPY

        data_entity = OTXDataEntity(
            tv_tensors.Image(torch.randn(3, 224, 224)),
            ImageInfo(0, (224, 224, 3), (224, 224, 3), (0, 0, 0), (1.0, 1.0)),
        )
        assert data_entity.image_type == ImageType.TV_IMAGE


class TestOTXBatchDataEntity:
    def test_collate_fn(self, mocker) -> None:
        mocker.patch.object(OTXDataEntity, "task", return_value="detection")
        mocker.patch.object(OTXBatchDataEntity, "task", return_value="detection")
        data_entities = [
            OTXDataEntity(
                tv_tensors.Image(torch.randn(3, 224, 224)),
                ImageInfo(0, (3, 224, 224), (3, 224, 224), (0, 0, 0), (1.0, 1.0)),
            ),
            OTXDataEntity(
                tv_tensors.Image(torch.randn(3, 224, 224)),
                ImageInfo(0, (3, 224, 224), (3, 224, 224), (0, 0, 0), (1.0, 1.0)),
            ),
            OTXDataEntity(
                tv_tensors.Image(torch.randn(3, 224, 224)),
                ImageInfo(0, (3, 224, 224), (3, 224, 224), (0, 0, 0), (1.0, 1.0)),
            ),
        ]

        data_batch = OTXBatchDataEntity.collate_fn(data_entities)
        assert len(data_batch.imgs_info) == len(data_batch.images)
