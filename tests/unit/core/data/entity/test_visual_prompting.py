# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of visual prompting data entity."""


import torch
from datumaro import Polygon
from otx.core.data.entity.base import ImageInfo, Points
from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchDataEntity,
    VisualPromptingDataEntity,
    ZeroShotVisualPromptingBatchDataEntity,
    ZeroShotVisualPromptingDataEntity,
)
from otx.core.types.task import OTXTaskType
from torch import LongTensor
from torchvision import tv_tensors


class TestVisualPromptingDataEntity:
    def test_task(self) -> None:
        data_entity = VisualPromptingDataEntity(
            image=tv_tensors.Image(torch.randn(3, 224, 224)),
            img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
            masks=tv_tensors.Mask(torch.randint(low=0, high=1, size=(224, 224))),
            labels=[LongTensor([1]), LongTensor([2])],
            polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
            bboxes=tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(224, 224)),
            points=Points(data=torch.Tensor([100, 100]), canvas_size=(224, 224)),
        )
        assert data_entity.task == OTXTaskType.VISUAL_PROMPTING


class TestVisualPromptingBatchDataEntity:
    def test_collate_fn(self) -> None:
        data_entities = [
            VisualPromptingDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                masks=tv_tensors.Mask(torch.randint(low=0, high=1, size=(224, 224))),
                labels=[LongTensor([1]), LongTensor([2])],
                polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
                bboxes=tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                points=Points(data=torch.Tensor([100, 100]), canvas_size=(224, 224)),
            ),
            VisualPromptingDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                masks=tv_tensors.Mask(torch.randint(low=0, high=1, size=(224, 224))),
                labels=[LongTensor([1]), LongTensor([2])],
                polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
                bboxes=tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                points=Points(data=torch.Tensor([100, 100]), canvas_size=(224, 224)),
            ),
            VisualPromptingDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                masks=tv_tensors.Mask(torch.randint(low=0, high=1, size=(224, 224))),
                labels=[LongTensor([1]), LongTensor([2])],
                polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
                bboxes=tv_tensors.BoundingBoxes(
                    data=torch.Tensor([0, 0, 50, 50]),
                    format="xywh",
                    canvas_size=(224, 224),
                ),
                points=Points(data=torch.Tensor([100, 100]), canvas_size=(224, 224)),
            ),
        ]

        data_batch = VisualPromptingBatchDataEntity.collate_fn(data_entities)
        assert len(data_batch.imgs_info) == len(data_batch.images)
        assert data_batch.task == OTXTaskType.VISUAL_PROMPTING


class TestZeroShotVisualPromptingDataEntity:
    def test_task(self) -> None:
        data_entity = ZeroShotVisualPromptingDataEntity(
            image=tv_tensors.Image(torch.randn(3, 224, 224)),
            img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
            masks=tv_tensors.Mask(torch.randint(low=0, high=1, size=(224, 224))),
            labels=[LongTensor([1]), LongTensor([2])],
            polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
            prompts=[
                tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(224, 224)),
                Points(data=torch.Tensor([100, 100]), canvas_size=(224, 224)),
            ],
        )
        assert data_entity.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING


class TestZeroShotVisualPromptingBatchDataEntity:
    def test_collate_fn(self) -> None:
        data_entities = [
            ZeroShotVisualPromptingDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                masks=tv_tensors.Mask(torch.randint(low=0, high=1, size=(224, 224))),
                labels=[LongTensor([1]), LongTensor([2])],
                polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
                prompts=[
                    tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(224, 224)),
                    Points(data=torch.Tensor([100, 100]), canvas_size=(224, 224)),
                ],
            ),
            ZeroShotVisualPromptingDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                masks=tv_tensors.Mask(torch.randint(low=0, high=1, size=(224, 224))),
                labels=[LongTensor([1]), LongTensor([2])],
                polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
                prompts=[
                    tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(224, 224)),
                    Points(data=torch.Tensor([100, 100]), canvas_size=(224, 224)),
                ],
            ),
            ZeroShotVisualPromptingDataEntity(
                image=tv_tensors.Image(torch.randn(3, 224, 224)),
                img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
                masks=tv_tensors.Mask(torch.randint(low=0, high=1, size=(224, 224))),
                labels=[LongTensor([1]), LongTensor([2])],
                polygons=[Polygon(points=[1, 1, 2, 2, 3, 3, 4, 4])],
                prompts=[
                    tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(224, 224)),
                    Points(data=torch.Tensor([100, 100]), canvas_size=(224, 224)),
                ],
            ),
        ]

        data_batch = ZeroShotVisualPromptingBatchDataEntity.collate_fn(data_entities)
        assert len(data_batch.imgs_info) == len(data_batch.images)
        assert data_batch.task == OTXTaskType.ZERO_SHOT_VISUAL_PROMPTING
