# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests of detection data transform."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.transform_libs.torchvision import (
    CachedMixUp,
    CachedMosaic,
    DecodeVideo,
    MinIoURandomCrop,
    Normalize,
    NumpytoTVTensor,
    PackVideo,
    Pad,
    PhotoMetricDistortion,
    RandomAffine,
    RandomFlip,
    Resize,
    YOLOXHSVRandomAug,
)
from otx.core.data.transform_libs.utils import overlap_bboxes, to_np_image
from torch import LongTensor, Tensor
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F  # noqa: N812


class MockFrame:
    data = np.ndarray([3, 10, 10])


class MockVideo:
    data = [MockFrame()] * 10

    def __getitem__(self, idx):
        return self.data[idx]

    def close(self):
        return


class TestDecodeVideo:
    def test_train_case(self):
        transform = DecodeVideo(test_mode=False)
        video = MockVideo()
        assert len(transform._transform(video, {})) == 8

        transform = DecodeVideo(test_mode=False, out_of_bound_opt="repeat_last")
        assert len(transform._transform(video, {})) == 8

    def test_eval_case(self):
        transform = DecodeVideo(test_mode=True)
        video = MockVideo()
        assert len(transform._transform(video, {})) == 8

        transform = DecodeVideo(test_mode=True, out_of_bound_opt="repeat_last")
        assert len(transform._transform(video, {})) == 8


class TestPackVideo:
    def test_forward(self):
        entity = ActionClsDataEntity(
            video=MockVideo(),
            image=[],
            img_info=ImageInfo(
                img_idx=0,
                img_shape=(0, 0),
                ori_shape=(0, 0),
                image_color_channel=None,
            ),
            labels=torch.LongTensor([0]),
        )
        transform = PackVideo()
        out = transform(entity)
        assert out.image == entity.video


@pytest.fixture()
def det_data_entity() -> DetDataEntity:
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

    def test_forward(self, min_iou_random_crop, det_data_entity) -> None:
        """Test forward."""
        results = min_iou_random_crop(deepcopy(det_data_entity))

        if (mode := min_iou_random_crop.mode) == 1:
            assert torch.equal(results.bboxes, det_data_entity.bboxes)
        else:
            patch = tv_tensors.wrap(torch.tensor([[0, 0, *results.img_info.img_shape]]), like=results.bboxes)
            ious = overlap_bboxes(patch, results.bboxes)
            assert torch.all(ious >= mode)
            assert results.image.shape[:2] == results.img_info.img_shape
            assert results.img_info.scale_factor is None


class TestResize:
    @pytest.fixture()
    def resize(self) -> Resize:
        return Resize(scale=(448, 448))  # (112, 224) -> (448, 448)

    @pytest.mark.parametrize(
        ("keep_ratio", "expected"),
        [
            (True, torch.tensor([[0.0, 0.0, 100.0, 100.0]])),
            (False, torch.tensor([[0.0, 0.0, 100.0, 200.0]])),
        ],
    )
    def test_forward(self, resize, det_data_entity, keep_ratio: bool, expected: Tensor) -> None:
        """Test forward."""
        resize.keep_ratio = keep_ratio
        det_data_entity.img_info.img_shape = resize.scale

        results = resize(deepcopy(det_data_entity))

        assert results.img_info.ori_shape == (112, 224)
        if keep_ratio:
            assert results.image.shape == (224, 448, 3)
            assert results.img_info.img_shape == (224, 448)
            assert results.img_info.scale_factor == (2.0, 2.0)
        else:
            assert results.image.shape == (448, 448, 3)
            assert results.img_info.img_shape == (448, 448)
            assert results.img_info.scale_factor == (2.0, 4.0)

        assert torch.all(results.bboxes.data == expected)

    def test_forward_without_bboxes(self, resize, det_data_entity) -> None:
        """Test forward."""
        resize.keep_ratio = True
        resize.transform_bbox = False  # set `transform_bbox` to False
        det_data_entity.img_info.img_shape = resize.scale

        results = resize(deepcopy(det_data_entity))

        assert results.img_info.ori_shape == (112, 224)
        assert results.image.shape == (224, 448, 3)
        assert results.img_info.img_shape == (224, 448)
        assert results.img_info.scale_factor == (2.0, 2.0)
        assert torch.all(results.bboxes.data == det_data_entity.bboxes.data)


class TestRandomFlip:
    @pytest.fixture()
    def random_flip(self) -> RandomFlip:
        return RandomFlip(prob=1.0)

    def test_forward(self, random_flip, det_data_entity) -> None:
        """Test forward."""
        results = random_flip.forward(deepcopy(det_data_entity))

        assert torch.all(F.to_image(results.image).flip(-1) == det_data_entity.image)

        bboxes_results = results.bboxes.clone()
        bboxes_results[..., 0] = results.img_info.img_shape[1] - results.bboxes[..., 2]
        bboxes_results[..., 2] = results.img_info.img_shape[1] - results.bboxes[..., 0]
        assert torch.all(bboxes_results == det_data_entity.bboxes)


class TestPhotoMetricDistortion:
    @pytest.fixture()
    def photo_metric_distortion(self) -> PhotoMetricDistortion:
        return PhotoMetricDistortion()

    def test_forward(self, photo_metric_distortion, det_data_entity) -> None:
        """Test forward."""
        results = photo_metric_distortion(deepcopy(det_data_entity))

        assert results.image.dtype == np.float32


class TestRandomAffine:
    @pytest.fixture()
    def random_affine(self) -> RandomAffine:
        return RandomAffine()

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_translate_ratio(self) -> None:
        RandomAffine(max_translate_ratio=1.5)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_scaling_ratio_range_inverse_order(self) -> None:
        RandomAffine(scaling_ratio_range=(1.5, 0.5))

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_scaling_ratio_range_zero_value(self) -> None:
        RandomAffine(scaling_ratio_range=(0, 0.5))

    def test_forward(self, random_affine, det_data_entity) -> None:
        """Test forward."""
        results = random_affine(deepcopy(det_data_entity))

        assert results.image.shape[:2] == (112, 224)
        assert results.labels.shape[0] == results.bboxes.shape[0]
        assert results.labels.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32
        assert results.img_info.img_shape == results.image.shape[:2]


class TestCachedMosaic:
    @pytest.fixture()
    def cached_mosaic(self) -> CachedMosaic:
        return CachedMosaic(random_pop=False, max_cached_images=20)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_img_scale(self) -> None:
        CachedMosaic(img_scale=640)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_probability(self) -> None:
        CachedMosaic(prob=1.5)

    def test_forward(self, cached_mosaic, det_data_entity) -> None:
        """Test forward."""
        cached_mosaic.mix_results = [deepcopy(det_data_entity)] * 3

        results = cached_mosaic(deepcopy(det_data_entity))

        assert results.image.shape[-2:] == (112, 224)
        assert results.labels.shape[0] == results.bboxes.shape[0]
        assert results.labels.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32
        assert results.img_info.img_shape == results.image.shape[-2:]


class TestCachedMixUp:
    @pytest.fixture()
    def cached_mixup(self) -> CachedMixUp:
        return CachedMixUp(ratio_range=(1.0, 1.0), prob=0.5, random_pop=False, max_cached_images=10)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_img_scale(self) -> None:
        CachedMixUp(img_scale=640)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_probability(self) -> None:
        CachedMosaic(prob=1.5)

    def test_forward(self, cached_mixup, det_data_entity) -> None:
        """Test forward."""
        cached_mixup.mix_results = [deepcopy(det_data_entity)]

        results = cached_mixup(deepcopy(det_data_entity))

        assert results.image.shape[-2:] == (112, 224)
        assert results.labels.shape[0] == results.bboxes.shape[0]
        assert results.labels.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32
        assert results.img_info.img_shape == results.image.shape[-2:]


class TestYOLOXHSVRandomAug:
    @pytest.fixture()
    def yolox_hsv_random_aug(self) -> YOLOXHSVRandomAug:
        return YOLOXHSVRandomAug()

    def test_forward(self, yolox_hsv_random_aug, det_data_entity) -> None:
        """Test forward."""
        results = yolox_hsv_random_aug(deepcopy(det_data_entity))

        assert results.image.shape[:2] == (112, 224)
        assert results.labels.shape[0] == results.bboxes.shape[0]
        assert results.labels.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32


class TestPad:
    def test_forward(self, det_data_entity) -> None:
        # test pad img/gt_masks with size
        transform = Pad(size=(200, 250))

        results = transform(deepcopy(det_data_entity))

        assert results.image.shape[:2] == (200, 250)

        # test pad img/gt_masks with size_divisor
        transform = Pad(size_divisor=11)

        results = transform(deepcopy(det_data_entity))

        assert results.image.shape[:2] == (121, 231)

        # test pad img/gt_masks with pad_to_square
        transform = Pad(pad_to_square=True)

        results = transform(deepcopy(det_data_entity))

        assert results.image.shape[:2] == (224, 224)

        # test pad img/gt_masks with pad_to_square and size_divisor
        transform = Pad(pad_to_square=True, size_divisor=11)

        results = transform(deepcopy(det_data_entity))

        # TODO (sungchul): check type
        assert results.image.shape[:2] == (231, 231)


class TestNormalize:
    def test_normalize(self, det_data_entity) -> None:
        det_data_entity.image = torch.ones((3, 2, 2)) * 2
        det_data_entity.image[1] *= 2
        det_data_entity.image[2] *= 3

        # test normalize
        img_norm_cfg = {"mean": [1, 2, 3], "std": [2, 2, 2], "to_rgb": False}
        transform = Normalize(**img_norm_cfg)

        results = transform(deepcopy(det_data_entity))

        assert np.all(
            results.image == np.array([[[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]], [[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]]]),
        )

        # test to_rgb=True
        img_norm_cfg = {"mean": [3, 2, 1], "std": [2, 2, 2], "to_rgb": True}
        transform = Normalize(**img_norm_cfg)

        results = transform(deepcopy(det_data_entity))

        assert np.all(
            results.image == np.array([[[1.5, 1.0, 0.5], [1.5, 1.0, 0.5]], [[1.5, 1.0, 0.5], [1.5, 1.0, 0.5]]]),
        )

    def test_repr(self):
        img_norm_cfg = {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "to_rgb": True}
        transform = Normalize(**img_norm_cfg)
        assert repr(transform) == ("Normalize(mean=[123.675 116.28  103.53 ], std=[58.395 57.12  57.375], to_rgb=True)")


class TestNumpytoTVTensor:
    def test_numpy_to_tvtensor(self, det_data_entity) -> None:
        det_data_entity.image = to_np_image(det_data_entity.image)

        # test
        transform = NumpytoTVTensor()

        results = transform(deepcopy(det_data_entity))

        assert isinstance(results.image, tv_tensors.Image)
