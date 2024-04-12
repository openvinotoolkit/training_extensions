# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of detection data transform."""

from __future__ import annotations

import pytest
from copy import deepcopy
import torch
from otx.core.data.transform_libs.torchvision import MinIoURandomCrop, Resize, RandomFlip
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.base import ImageInfo
from torch import LongTensor
from torchvision import tv_tensors


@pytest.fixture()
def data_entity() -> DetDataEntity:
    return DetDataEntity(
        image=tv_tensors.Image(torch.randint(low=0, high=256, size=(3, 112, 224), dtype=torch.uint8)),
        img_info=ImageInfo(img_idx=0, img_shape=(112, 224), ori_shape=(112, 224)),
        bboxes=tv_tensors.BoundingBoxes(data=torch.Tensor([0, 0, 50, 50]), format="xywh", canvas_size=(112, 224)),
        labels=LongTensor([1]),
    )


class TestResize:
    @pytest.fixture()
    def resize(self) -> Resize:
        return Resize(scale=(448, 448))

    @pytest.mark.parametrize("keep_ratio", [True, False])
    def test_resize_img(self, resize, data_entity, keep_ratio: bool) -> None:
        """Test _resize_img."""
        resize.keep_ratio = keep_ratio
        data_entity.img_info.img_shape = resize.scale

        results = resize._resize_img(data_entity)

        assert results.img_info.ori_shape == (112, 224)
        if keep_ratio:
            assert results.image.shape == (3, 224, 448)
            assert results.img_info.img_shape == (224, 448)
            assert results.img_info.scale_factor == (2., 2.)
        else:
            assert results.image.shape == (3, 448, 448)
            assert results.img_info.img_shape == (448, 448)
            assert results.img_info.scale_factor == (2., 4.)

    @pytest.mark.parametrize(
        ("scale_factor", "expected"),
        [
            ((2., 2.), torch.tensor([[0., 0., 100., 100.]])),
            ((2., 4.), torch.tensor([[0., 0., 200., 100.]])), 
        ]
    )
    def test_resize_bboxes(self, resize, data_entity, scale_factor: tuple[float, float], expected: Tensor) -> None:
        """Test _resize_bboxes."""
        data_entity.img_info.scale_factor = scale_factor

        results = resize._resize_bboxes(data_entity)

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
