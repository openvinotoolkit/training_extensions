# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests of detection data transform."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from datumaro import Polygon
from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.transform_libs.torchvision import (
    CachedMixUp,
    CachedMosaic,
    DecodeVideo,
    MinIoURandomCrop,
    PackVideo,
    Pad,
    PhotoMetricDistortion,
    RandomAffine,
    RandomFlip,
    Resize,
    YOLOXHSVRandomAug,
)
from otx.core.data.transform_libs.utils import overlap_bboxes
from torch import LongTensor
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
        return Resize(scale=(128, 96))  # (64, 64) -> (96, 128)

    @pytest.mark.parametrize(
        ("keep_ratio", "expected_shape", "expected_scale_factor"),
        [
            (True, (96, 96), (1.5, 1.5)),
            (False, (96, 128), (2.0, 1.5)),
        ],
    )
    def test_forward_only_image(
        self,
        resize,
        fxt_det_data_entity,
        keep_ratio: bool,
        expected_shape: tuple,
        expected_scale_factor: tuple,
    ) -> None:
        """Test forward only image."""
        resize.keep_ratio = keep_ratio
        resize.transform_bbox = False
        resize.transform_mask = False
        entity = deepcopy(fxt_det_data_entity[0])
        entity.image = entity.image.transpose(1, 2, 0)

        results = resize(entity)

        assert results.img_info.ori_shape == (64, 64)
        if keep_ratio:
            assert results.image.shape[:2] == expected_shape
            assert results.img_info.img_shape == expected_shape
            assert results.img_info.scale_factor == expected_scale_factor
        else:
            assert results.image.shape[:2] == expected_shape
            assert results.img_info.img_shape == expected_shape
            assert results.img_info.scale_factor == expected_scale_factor

        assert torch.all(results.bboxes.data == fxt_det_data_entity[0].bboxes.data)

    @pytest.mark.parametrize(
        ("keep_ratio", "expected_shape"),
        [
            (True, (96, 96)),
            (False, (96, 128)),
        ],
    )
    def test_forward_bboxes_masks_polygons(
        self,
        resize,
        fxt_inst_seg_data_entity,
        keep_ratio: bool,
        expected_shape: tuple,
    ) -> None:
        """Test forward with bboxes, masks, and polygons."""
        resize.transform_bbox = True
        resize.transform_mask = True
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        entity.image = entity.image.transpose(1, 2, 0)

        resize.keep_ratio = keep_ratio
        results = resize(entity)

        assert results.image.shape[:2] == expected_shape
        assert results.img_info.img_shape == expected_shape
        assert torch.all(
            results.bboxes == fxt_inst_seg_data_entity[0].bboxes * torch.tensor(results.img_info.scale_factor * 2),
        )
        assert results.masks.shape[1:] == expected_shape
        assert all(
            [  # noqa: C419
                np.all(
                    np.array(rp.points).reshape(-1, 2)
                    == np.array(fp.points).reshape(-1, 2) * np.array([results.img_info.scale_factor]),
                )
                for rp, fp in zip(results.polygons, fxt_inst_seg_data_entity[0].polygons)
            ],
        )


class TestRandomFlip:
    @pytest.fixture()
    def random_flip(self) -> RandomFlip:
        return RandomFlip(prob=1.0)

    def test_forward(self, random_flip, fxt_inst_seg_data_entity) -> None:
        """Test forward."""
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        entity.image = entity.image.transpose(1, 2, 0)

        results = random_flip.forward(entity)

        # test image
        assert np.all(F.to_image(results.image).flip(-1).numpy() == fxt_inst_seg_data_entity[0].image)

        # test bboxes
        bboxes_results = results.bboxes.clone()
        bboxes_results[..., 0] = results.img_info.img_shape[1] - results.bboxes[..., 2]
        bboxes_results[..., 2] = results.img_info.img_shape[1] - results.bboxes[..., 0]
        assert torch.all(bboxes_results == fxt_inst_seg_data_entity[0].bboxes)

        # test masks
        assert torch.all(tv_tensors.Mask(results.masks).flip(-1) == fxt_inst_seg_data_entity[0].masks)

        # test polygons
        def revert_hflip(polygon: list[float], width: int) -> list[float]:
            p = np.asarray(polygon.points)
            p[0::2] = width - p[0::2]
            return p.tolist()

        width = results.img_info.img_shape[1]
        polygons_results = deepcopy(results.polygons)
        polygons_results = [Polygon(points=revert_hflip(polygon, width)) for polygon in polygons_results]
        assert polygons_results == fxt_inst_seg_data_entity[0].polygons


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
        return CachedMosaic(img_scale=(128, 128), random_pop=False, max_cached_images=20)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_img_scale(self) -> None:
        CachedMosaic(img_scale=640)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_probability(self) -> None:
        CachedMosaic(prob=1.5)

    def test_forward_pop_small_cache(self, cached_mosaic, fxt_inst_seg_data_entity) -> None:
        """Test forward for popping cache."""
        cached_mosaic.max_cached_images = 4
        cached_mosaic.results_cache = [fxt_inst_seg_data_entity[0]] * cached_mosaic.max_cached_images

        # 4 -> 5 thru append -> 4 thru pop -> return due to small cache
        results = cached_mosaic(deepcopy(fxt_inst_seg_data_entity[0]))

        # check pop
        assert len(cached_mosaic.results_cache) == cached_mosaic.max_cached_images

        # check small cache
        assert np.all(results.image == fxt_inst_seg_data_entity[0].image)
        assert torch.all(results.bboxes == fxt_inst_seg_data_entity[0].bboxes)

    def test_forward(self, cached_mosaic, fxt_inst_seg_data_entity) -> None:
        """Test forward."""
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        entity.image = entity.image.transpose(1, 2, 0)
        cached_mosaic.results_cache = [entity] * 4
        cached_mosaic.prob = 1.0

        results = cached_mosaic(deepcopy(entity))

        assert results.image.shape[:2] == (256, 256)
        assert results.labels.shape[0] == results.bboxes.shape[0]
        assert results.labels.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32
        assert results.img_info.img_shape == results.image.shape[:2]
        assert results.masks.shape[1:] == (256, 256)
        assert len(results.polygons) == 4


class TestCachedMixUp:
    @pytest.fixture()
    def cached_mixup(self) -> CachedMixUp:
        return CachedMixUp(ratio_range=(1.0, 1.0), prob=1.0, random_pop=False, max_cached_images=10)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_img_scale(self) -> None:
        CachedMixUp(img_scale=640)

    @pytest.mark.xfail(raises=AssertionError)
    def test_init_invalid_probability(self) -> None:
        CachedMosaic(prob=1.5)

    def test_forward_pop_small_cache(self, cached_mixup, fxt_inst_seg_data_entity) -> None:
        """Test forward for popping cache."""
        cached_mixup.max_cached_images = 1  # force to set to 1 for this test
        cached_mixup.results_cache = [fxt_inst_seg_data_entity[0]] * cached_mixup.max_cached_images

        # 1 -> 2 thru append -> 1 thru pop -> return due to small cache
        results = cached_mixup(deepcopy(fxt_inst_seg_data_entity[0]))

        # check pop
        assert len(cached_mixup.results_cache) == cached_mixup.max_cached_images

        # check small cache
        assert np.all(results.image == fxt_inst_seg_data_entity[0].image)
        assert torch.all(results.bboxes == fxt_inst_seg_data_entity[0].bboxes)

    def test_forward(self, cached_mixup, fxt_inst_seg_data_entity) -> None:
        """Test forward."""
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        entity.image = entity.image.transpose(1, 2, 0)
        cached_mixup.results_cache = [entity]
        cached_mixup.prob = 1.0
        cached_mixup.flip_ratio = 0.0

        results = cached_mixup(deepcopy(entity))

        assert results.image.shape[:2] == (64, 64)
        assert results.labels.shape[0] == results.bboxes.shape[0]
        assert results.labels.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32
        assert results.img_info.img_shape == results.image.shape[:2]
        assert results.masks.shape[1:] == (64, 64)
        assert len(results.polygons) == 1


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
    def test_forward(self, fxt_inst_seg_data_entity) -> None:
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        entity.image = entity.image.transpose(1, 2, 0)

        # test pad img/masks with size
        transform = Pad(size=(96, 128), transform_mask=True)

        results = transform(deepcopy(entity))

        assert results.image.shape[:2] == (96, 128)
        assert results.masks.shape[1:] == (96, 128)

        # test pad img/masks with size_divisor
        transform = Pad(size_divisor=11, transform_mask=True)

        results = transform(deepcopy(entity))

        # (64, 64) -> (66, 66)
        assert results.image.shape[:2] == (66, 66)
        assert results.masks.shape[1:] == (66, 66)

        # test pad img/masks with pad_to_square
        _transform = Pad(size=(96, 128), transform_mask=True)
        entity = _transform(deepcopy(entity))
        transform = Pad(pad_to_square=True, transform_mask=True)

        results = transform(deepcopy(entity))

        assert results.image.shape[:2] == (128, 128)
        assert results.masks.shape[1:] == (128, 128)

        # test pad img/masks with pad_to_square and size_divisor
        _transform = Pad(size=(96, 128), transform_mask=True)
        entity = _transform(deepcopy(entity))
        transform = Pad(pad_to_square=True, size_divisor=11, transform_mask=True)

        results = transform(deepcopy(entity))

        assert results.image.shape[:2] == (132, 132)
        assert results.masks.shape[1:] == (132, 132)
