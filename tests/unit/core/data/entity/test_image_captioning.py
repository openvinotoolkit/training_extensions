# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.image_captioning import (
    ImageCaptionBatchDataEntity,
    ImageCaptionDataEntity,
)
from otx.core.types.task import OTXTaskType
from torchvision import tv_tensors


class TestImageCaptionDataEntity:
    def test_creation(self) -> None:
        data_entity = ImageCaptionDataEntity(
            image=tv_tensors.Image(torch.randn(3, 224, 224)),
            img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
            captions=["caption1", "caption2"],
        )
        assert data_entity.task == OTXTaskType.IMAGE_CAPTIONING


class TestImageCaptionBatchDataEntity:
    def test_collate_fn(self) -> None:
        data_entities = [
            ImageCaptionDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                captions=["caption1", "caption2"],
            ),
            ImageCaptionDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                captions=["caption1", "caption2"],
            ),
            ImageCaptionDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                captions=["caption1", "caption2"],
            ),
        ]

        data_batch = ImageCaptionBatchDataEntity.collate_fn(data_entities)
        assert len(data_batch.imgs_info) == len(data_batch.images)
        assert data_batch.task == OTXTaskType.IMAGE_CAPTIONING
        assert len(data_batch.captions) == 3
