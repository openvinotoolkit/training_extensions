# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of detection data transform."""

from __future__ import annotations

import pytest
from copy import deepcopy
import torch
from otx.core.data.transform_libs.torchvision import MinIoURandomCrop, Resize, RandomFlip, PhotoMetricDistortion
from otx.core.data.transform_libs.utils import overlap_bboxes
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.base import ImageInfo
from torch import LongTensor, Tensor
from torchvision import tv_tensors


@pytest.fixture()
def data_entity() -> DetDataEntity:
    return DetDataEntity(
        image=tv_tensors.Image(torch.randint(low=0, high=256, size=(3, 112, 224), dtype=torch.uint8)),
        img_info=ImageInfo(img_idx=0, img_shape=(112, 224), ori_shape=(112, 224)),
        bboxes=tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(112, 224)),
        labels=LongTensor([1]),
    )


class TestMinIoURandomCrop:
    @pytest.fixture()
    def min_iou_random_crop(self) -> MinIoURandomCrop:
        return MinIoURandomCrop()

    def test_forward(self, min_iou_random_crop, data_entity) -> None:
        """Test forward."""
        results = min_iou_random_crop(deepcopy(data_entity))

        if (mode := min_iou_random_crop.mode) == 1:
            assert torch.equal(results.bboxes, data_entity.bboxes)
        else:
            patch = tv_tensors.wrap(
                torch.tensor([[0, 0, *results.img_info.img_shape]]),
                like=results.bboxes)
            ious = overlap_bboxes(patch, results.bboxes)
            assert torch.all(ious >= mode)
            assert results.image.shape[-2:] == results.img_info.img_shape
            assert results.img_info.scale_factor is None


class TestResize:
    @pytest.fixture()
    def resize(self) -> Resize:
        return Resize(scale=(448, 448)) # (112, 224) -> (448, 448)

    @pytest.mark.parametrize(
        ("keep_ratio", "expected"), 
        [
            (True, torch.tensor([[0., 0., 100., 100.]])),
            (False, torch.tensor([[0., 0., 100., 200.]]))
        ]
    )
    def test_forward(self, resize, data_entity, keep_ratio: bool, expected: Tensor) -> None:
        """Test forward."""
        resize.keep_ratio = keep_ratio
        data_entity.img_info.img_shape = resize.scale

        results = resize(deepcopy(data_entity))

        assert results.img_info.ori_shape == (112, 224)
        if keep_ratio:
            assert results.image.shape == (3, 224, 448)
            assert results.img_info.img_shape == (224, 448)
            assert results.img_info.scale_factor == (2., 2.)
        else:
            assert results.image.shape == (3, 448, 448)
            assert results.img_info.img_shape == (448, 448)
            assert results.img_info.scale_factor == (2., 4.)

        assert torch.all(results.bboxes.data == expected)


class TestRandomFlip:
    @pytest.fixture()
    def random_flip(self) -> RandomFlip:
        return RandomFlip(prob=1.)

    def test_forward(self, random_flip, data_entity) -> None:
        """Test forward."""
        results = random_flip.forward(deepcopy(data_entity))

        assert torch.all(results.image.flip(-1) == data_entity.image)

        bboxes_results = results.bboxes.clone()
        bboxes_results[..., 0] = results.img_info.img_shape[1] - results.bboxes[..., 2]
        bboxes_results[..., 2] = results.img_info.img_shape[1] - results.bboxes[..., 0]
        assert torch.all(bboxes_results == data_entity.bboxes)


class TestPhotoMetricDistortion:
    @pytest.fixture()
    def photo_metric_distortion(self) -> PhotoMetricDistortion:
        return PhotoMetricDistortion()

    def test_forward(self, photo_metric_distortion, data_entity) -> None:
        """Test forward."""
        results = photo_metric_distortion(deepcopy(data_entity))

        assert results.image.dtype == torch.float32
