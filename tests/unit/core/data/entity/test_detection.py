# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of detection data entity."""

import torch
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetBatchDataEntity, DetDataEntity
from otx.core.types.task import OTXTaskType
from torch import LongTensor
from torchvision import tv_tensors


class TestDetDataEntity:
    def test_task(self) -> None:
        data_entity = DetDataEntity(
            tv_tensors.Image(torch.randn(3, 224, 224)),
            ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
            tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(224, 224)),
            LongTensor([1]),
        )
        assert data_entity.task == OTXTaskType.DETECTION


class TestDetBatchDataEntity:
    def test_collate_fn(self) -> None:
        data_entities = [
            DetDataEntity(
                tv_tensors.Image(torch.randn(3, 224, 224)),
                ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                LongTensor([1]),
            ),
            DetDataEntity(
                tv_tensors.Image(torch.randn(3, 224, 224)),
                ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                LongTensor([1]),
            ),
            DetDataEntity(
                tv_tensors.Image(torch.randn(3, 224, 224)),
                ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                LongTensor([1]),
            ),
        ]

        data_batch = DetBatchDataEntity.collate_fn(data_entities)
        assert len(data_batch.imgs_info) == len(data_batch.images)
        assert data_batch.task == OTXTaskType.DETECTION
