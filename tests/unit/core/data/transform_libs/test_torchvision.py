# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) OpenMMLab. All rights reserved.
"""Unit tests of detection data transform."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from datumaro import Polygon
from otx.core.data.entity.action_classification import ActionClsDataEntity
from otx.core.data.entity.base import BboxInfo, ImageInfo, OTXDataEntity, VideoInfo
from otx.core.data.entity.detection import DetBatchDataEntity, DetDataEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegDataEntity
from otx.core.data.entity.keypoint_detection import KeypointDetDataEntity
from otx.core.data.transform_libs.torchvision import (
    CachedMixUp,
    CachedMosaic,
    Compose,
    DecodeVideo,
    FilterAnnotations,
    GetBBoxCenterScale,
    MinIoURandomCrop,
    PackVideo,
    Pad,
    PhotoMetricDistortion,
    RandomAffine,
    RandomCrop,
    RandomFlip,
    RandomResize,
    Resize,
    TopdownAffine,
    YOLOXHSVRandomAug,
)
from otx.core.data.transform_libs.utils import overlap_bboxes
from torch import LongTensor
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F  # noqa: N812

if TYPE_CHECKING:
    from otx.core.data.entity.classification import MulticlassClsBatchDataEntity, MulticlassClsDataEntity


class MockFrame:
    data = np.ndarray([10, 10, 3], dtype=np.uint8)


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
            video_info=VideoInfo(),
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

    def test_forward(self, min_iou_random_crop: MinIoURandomCrop, det_data_entity: DetDataEntity) -> None:
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
        return Resize(scale=(128, 96))  # (64, 64) -> (128, 96)

    @pytest.mark.parametrize(
        ("keep_ratio", "expected_shape", "expected_scale_factor"),
        [
            (True, (96, 96), (1.5, 1.5)),
            (False, (128, 96), (2.0, 1.5)),
        ],
    )
    @pytest.mark.parametrize("is_array", [True, False])
    def test_forward_only_image(
        self,
        resize: Resize,
        fxt_det_data_entity: tuple[tuple, DetDataEntity, DetBatchDataEntity],
        keep_ratio: bool,
        is_array: bool,
        expected_shape: tuple,
        expected_scale_factor: tuple,
    ) -> None:
        """Test forward only image."""
        resize.keep_ratio = keep_ratio
        resize.transform_bbox = False
        resize.transform_mask = False
        entity = deepcopy(fxt_det_data_entity[0])
        if is_array:
            entity.image = entity.image.transpose(1, 2, 0)
        else:
            entity.image = torch.as_tensor(entity.image)

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
        ("keep_ratio", "expected_shape", "expected_scale_factor"),
        [
            (True, (96, 96), (1.5, 1.5)),
            (False, (128, 96), (2.0, 1.5)),
        ],
    )
    @pytest.mark.parametrize("is_array", [True, False])
    def test_forward_only_image_with_list_of_images(
        self,
        resize: Resize,
        fxt_det_data_entity: tuple[tuple, DetDataEntity, DetBatchDataEntity],
        keep_ratio: bool,
        is_array: bool,
        expected_shape: tuple,
        expected_scale_factor: tuple,
    ) -> None:
        """Test forward only image."""
        resize.keep_ratio = keep_ratio
        resize.transform_bbox = False
        resize.transform_mask = False
        entity = deepcopy(fxt_det_data_entity[0])
        if is_array:
            entity.image = entity.image.transpose(1, 2, 0)
        else:
            entity.image = torch.as_tensor(entity.image)

        entity.image = [entity.image, entity.image]

        results = resize(entity)

        assert results.img_info.ori_shape == (64, 64)
        if keep_ratio:
            assert results.image[0].shape[:2] == expected_shape
            assert results.image[1].shape[:2] == expected_shape
            assert results.img_info.img_shape == expected_shape
            assert results.img_info.scale_factor == expected_scale_factor
        else:
            assert results.image[0].shape[:2] == expected_shape
            assert results.image[1].shape[:2] == expected_shape
            assert results.img_info.img_shape == expected_shape
            assert results.img_info.scale_factor == expected_scale_factor

        assert torch.all(results.bboxes.data == fxt_det_data_entity[0].bboxes.data)

    @pytest.mark.parametrize(
        ("keep_ratio", "expected_shape"),
        [
            (True, (96, 96)),
            (False, (128, 96)),
        ],
    )
    def test_forward_bboxes_masks_polygons(
        self,
        resize: Resize,
        fxt_inst_seg_data_entity: tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity],
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
            results.bboxes
            == fxt_inst_seg_data_entity[0].bboxes * torch.tensor(results.img_info.scale_factor[::-1] * 2),
        )
        assert results.masks.shape[1:] == expected_shape
        assert all(
            [  # noqa: C419
                np.all(
                    np.array(rp.points).reshape(-1, 2)
                    == np.array(fp.points).reshape(-1, 2) * np.array([results.img_info.scale_factor[::-1]]),
                )
                for rp, fp in zip(results.polygons, fxt_inst_seg_data_entity[0].polygons)
            ],
        )


class TestRandomFlip:
    @pytest.fixture()
    def random_flip(self) -> RandomFlip:
        return RandomFlip(prob=1.0)

    @pytest.mark.parametrize("is_array", [True, False])
    def test_forward(
        self,
        random_flip: RandomFlip,
        fxt_inst_seg_data_entity: tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity],
        is_array: bool,
    ) -> None:
        """Test forward."""
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        if is_array:
            entity.image = entity.image.transpose(1, 2, 0)
        else:
            entity.image = torch.as_tensor(entity.image)

        results = random_flip.forward(entity)

        # test image
        assert np.all(F.to_image(results.image.copy()).flip(-1).numpy() == fxt_inst_seg_data_entity[0].image)

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

    @pytest.mark.parametrize("is_array", [True, False])
    def test_forward_with_list_of_images(
        self,
        random_flip: RandomFlip,
        fxt_multi_class_cls_data_entity: tuple[
            MulticlassClsDataEntity,
            MulticlassClsBatchDataEntity,
            MulticlassClsBatchDataEntity,
        ],
        is_array: bool,
    ) -> None:
        entity = deepcopy(fxt_multi_class_cls_data_entity[0])
        if is_array:
            entity.image = entity.image.transpose(1, 2, 0)
        else:
            entity.image = torch.as_tensor(entity.image)

        entity.image = [entity.image, entity.image]

        results = random_flip.forward(entity)

        # test image
        for img in results.image:
            assert np.all(F.to_image(img.copy()).flip(-1).numpy() == fxt_multi_class_cls_data_entity[0].image)


class TestPhotoMetricDistortion:
    @pytest.fixture()
    def photo_metric_distortion(self) -> PhotoMetricDistortion:
        return PhotoMetricDistortion()

    def test_forward(self, photo_metric_distortion: PhotoMetricDistortion, det_data_entity: DetDataEntity) -> None:
        """Test forward."""
        results = photo_metric_distortion(deepcopy(det_data_entity))

        assert results.image.dtype == np.float32


class TestRandomAffine:
    @pytest.fixture()
    def random_affine(self) -> RandomAffine:
        return RandomAffine()

    def test_init_invalid_translate_ratio(self) -> None:
        with pytest.raises(AssertionError):
            RandomAffine(max_translate_ratio=1.5)

    def test_init_invalid_scaling_ratio_range_inverse_order(self) -> None:
        with pytest.raises(AssertionError):
            RandomAffine(scaling_ratio_range=(1.5, 0.5))

    def test_init_invalid_scaling_ratio_range_zero_value(self) -> None:
        with pytest.raises(AssertionError):
            RandomAffine(scaling_ratio_range=(0, 0.5))

    def test_forward(self, random_affine: RandomAffine, det_data_entity: DetDataEntity) -> None:
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

    def test_init_invalid_img_scale(self) -> None:
        with pytest.raises(AssertionError):
            CachedMosaic(img_scale=640)

    def test_init_invalid_probability(self) -> None:
        with pytest.raises(AssertionError):
            CachedMosaic(prob=1.5)

    def test_forward_pop_small_cache(
        self,
        cached_mosaic: CachedMosaic,
        fxt_inst_seg_data_entity: tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity],
    ) -> None:
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

    def test_forward(
        self,
        cached_mosaic: CachedMosaic,
        fxt_inst_seg_data_entity: tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity],
    ) -> None:
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


class TestCachedMixUp:
    @pytest.fixture()
    def cached_mixup(self) -> CachedMixUp:
        return CachedMixUp(ratio_range=(1.0, 1.0), prob=1.0, random_pop=False, max_cached_images=10)

    def test_init_invalid_img_scale(self) -> None:
        with pytest.raises(AssertionError):
            CachedMixUp(img_scale=640)

    def test_init_invalid_probability(self) -> None:
        with pytest.raises(AssertionError):
            CachedMosaic(prob=1.5)

    def test_forward_pop_small_cache(
        self,
        cached_mixup: CachedMixUp,
        fxt_inst_seg_data_entity: tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity],
    ) -> None:
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

    def test_forward(
        self,
        cached_mixup: CachedMixUp,
        fxt_inst_seg_data_entity: tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity],
    ) -> None:
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


class TestYOLOXHSVRandomAug:
    @pytest.fixture()
    def yolox_hsv_random_aug(self) -> YOLOXHSVRandomAug:
        return YOLOXHSVRandomAug()

    def test_forward(self, yolox_hsv_random_aug: YOLOXHSVRandomAug, det_data_entity: DetDataEntity) -> None:
        """Test forward."""
        results = yolox_hsv_random_aug(deepcopy(det_data_entity))

        assert results.image.shape[:2] == (112, 224)
        assert results.labels.shape[0] == results.bboxes.shape[0]
        assert results.labels.dtype == torch.int64
        assert results.bboxes.dtype == torch.float32


class TestPad:
    def test_forward(
        self,
        fxt_inst_seg_data_entity: tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity],
    ) -> None:
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


class TestRandomResize:
    def test_init(self):
        transform = RandomResize((224, 224), (1.0, 2.0))
        assert transform.scale == (224, 224)

    def test_repr(self):
        transform = RandomResize((224, 224), (1.0, 2.0))
        transform_str = str(transform)
        assert isinstance(transform_str, str)

    def test_forward(self, fxt_inst_seg_data_entity: tuple[tuple, InstanceSegDataEntity, InstanceSegBatchDataEntity]):
        entity = deepcopy(fxt_inst_seg_data_entity[0])
        entity.image = entity.image.transpose(1, 2, 0)

        # choose target scale from init when override is True
        transform = RandomResize((224, 224), (1.0, 2.0))

        results = transform(deepcopy(entity))

        assert results.img_info.img_shape[0] >= 224
        assert results.img_info.img_shape[0] <= 448
        assert results.img_info.img_shape[1] >= 224
        assert results.img_info.img_shape[1] <= 448

        # keep ratio is True
        transform = RandomResize((224, 224), (1.0, 2.0), keep_ratio=True, transform_bbox=True, transform_mask=True)

        results = transform(deepcopy(entity))

        assert results.image.shape[0] >= 224
        assert results.image.shape[0] <= 448
        assert results.image.shape[1] >= 224
        assert results.image.shape[1] <= 448
        assert results.img_info.img_shape[0] >= 224
        assert results.img_info.img_shape[0] <= 448
        assert results.img_info.img_shape[1] >= 224
        assert results.img_info.img_shape[1] <= 448
        assert results.img_info.scale_factor[0] == results.img_info.scale_factor[1]
        assert results.bboxes[0, 2] == entity.bboxes[0, 2] * results.img_info.scale_factor[0]
        assert results.bboxes[0, 3] == entity.bboxes[0, 3] * results.img_info.scale_factor[1]
        assert results.masks.shape[1] >= 224
        assert results.masks.shape[1] <= 448
        assert results.masks.shape[2] >= 224
        assert results.masks.shape[2] <= 448

        # keep ratio is False
        transform = RandomResize((224, 224), (1.0, 2.0), keep_ratio=False, transform_bbox=True, transform_mask=True)

        results = transform(deepcopy(entity))

        # choose target scale from init when override is False and scale is a list of tuples
        transform = RandomResize([(448, 224), (224, 112)], keep_ratio=False, transform_bbox=True, transform_mask=True)

        results = transform(deepcopy(entity))

        assert results.img_info.img_shape[1] >= 112
        assert results.img_info.img_shape[1] <= 224
        assert results.img_info.img_shape[0] >= 224
        assert results.img_info.img_shape[0] <= 448

        # the type of scale is invalid in init
        with pytest.raises(NotImplementedError):
            RandomResize([(448, 224), [224, 112]], keep_ratio=True)(deepcopy(entity))


class TestRandomCrop:
    @pytest.fixture()
    def entity(self) -> OTXDataEntity:
        return OTXDataEntity(
            image=np.random.randint(0, 255, size=(24, 32), dtype=np.int32),
            img_info=ImageInfo(img_idx=0, img_shape=(24, 32), ori_shape=(24, 32)),
        )

    @pytest.fixture()
    def det_entity(self) -> DetDataEntity:
        return DetDataEntity(
            image=np.random.randint(0, 255, size=(10, 10), dtype=np.uint8),
            img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
            bboxes=tv_tensors.BoundingBoxes(
                np.array([[0, 0, 7, 7], [2, 3, 9, 9]], dtype=np.float32),
                format="xyxy",
                canvas_size=(10, 10),
            ),
            labels=torch.LongTensor([0, 1]),
        )

    @pytest.fixture()
    def iseg_entity(self) -> InstanceSegDataEntity:
        return InstanceSegDataEntity(
            image=np.random.randint(0, 255, size=(10, 10), dtype=np.uint8),
            img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
            bboxes=tv_tensors.BoundingBoxes(
                np.array([[0, 0, 7, 7], [2, 3, 9, 9]], dtype=np.float32),
                format="xyxy",
                canvas_size=(10, 10),
            ),
            labels=torch.LongTensor([0, 1]),
            masks=tv_tensors.Mask(np.zeros((2, 10, 10), np.uint8)),
            polygons=[Polygon(points=[0, 0, 0, 7, 7, 7, 7, 0]), Polygon(points=[2, 3, 2, 9, 9, 9, 9, 3])],
        )

    def test_init_invalid_crop_type(self) -> None:
        # test invalid crop_type
        with pytest.raises(ValueError, match="Invalid crop_type"):
            RandomCrop(crop_size=(10, 10), crop_type="unknown")

    @pytest.mark.parametrize("crop_type", ["absolute", "absolute_range"])
    @pytest.mark.parametrize("crop_size", [(0, 0), (0, 1), (1, 0)])
    def test_init_invalid_value(self, crop_type: str, crop_size: tuple[int, int]) -> None:
        # test h > 0 and w > 0
        with pytest.raises(AssertionError):
            RandomCrop(crop_size=crop_size, crop_type=crop_type)

    @pytest.mark.parametrize("crop_type", ["absolute", "absolute_range"])
    @pytest.mark.parametrize("crop_size", [(1.0, 1), (1, 1.0), (1.0, 1.0)])
    def test_init_invalid_type(self, crop_type: str, crop_size: tuple[int, int]) -> None:
        # test type(h) = int and type(w) = int
        with pytest.raises(AssertionError):
            RandomCrop(crop_size=crop_size, crop_type=crop_type)

    def test_init_invalid_size(self) -> None:
        # test crop_size[0] <= crop_size[1]
        with pytest.raises(AssertionError):
            RandomCrop(crop_size=(10, 5), crop_type="absolute_range")

    @pytest.mark.parametrize("crop_type", ["relative_range", "relative"])
    @pytest.mark.parametrize("crop_size", [(0, 1), (1, 0), (1.1, 0.5), (0.5, 1.1)])
    def test_init_invalid_range(self, crop_type: str, crop_size: tuple[int | float]) -> None:
        # test h in (0, 1] and w in (0, 1]
        with pytest.raises(AssertionError):
            RandomCrop(crop_size=crop_size, crop_type=crop_type)

    @pytest.mark.parametrize(("crop_type", "crop_size"), [("relative", (0.5, 0.5)), ("absolute", (12, 16))])
    def test_forward_relative_absolute(self, entity, crop_type: str, crop_size: tuple[float | int]) -> None:
        # test relative and absolute crop
        transform = RandomCrop(crop_size=crop_size, crop_type=crop_type)
        target_shape = (12, 16)

        results = transform(deepcopy(entity))

        assert results.image.shape[:2] == target_shape

    def test_forward_absolute_range(self, entity) -> None:
        # test absolute_range crop
        transform = RandomCrop(crop_size=(10, 20), crop_type="absolute_range")

        results = transform(deepcopy(entity))

        h, w = results.image.shape
        assert 10 <= w <= 20
        assert 10 <= h <= 20
        assert results.img_info.img_shape == results.image.shape[:2]

    def test_forward_relative_range(self, entity) -> None:
        # test relative_range crop
        transform = RandomCrop(crop_size=(0.9, 0.8), crop_type="relative_range")

        results = transform(deepcopy(entity))

        h, w = results.image.shape
        assert 24 * 0.9 <= h <= 24
        assert 32 * 0.8 <= w <= 32
        assert results.img_info.img_shape == results.image.shape[:2]

    def test_forward_bboxes_labels_masks_polygons(self, iseg_entity) -> None:
        # test with bboxes, labels, masks, and polygons
        transform = RandomCrop(crop_size=(7, 5), allow_negative_crop=False, recompute_bbox=False, bbox_clip_border=True)

        results = transform(deepcopy(iseg_entity))

        assert results.image.shape[:2] == (7, 5)
        assert results.bboxes.shape[0] == 2
        assert results.labels.shape[0] == 2
        assert results.masks.shape[0] == 2
        assert results.masks.shape[1:] == (7, 5)
        assert results.img_info.img_shape == results.image.shape[:2]

    def test_forward_recompute_bbox_from_mask(self, iseg_entity) -> None:
        # test recompute_bbox = True
        iseg_entity.bboxes = tv_tensors.wrap(torch.tensor([[0.1, 0.1, 0.2, 0.2]]), like=iseg_entity.bboxes)
        iseg_entity.labels = torch.LongTensor([0])
        iseg_entity.polygons = []
        target_gt_bboxes = np.zeros((1, 4), dtype=np.float32)
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=True,
        )

        results = transform(deepcopy(iseg_entity))

        assert np.all(results.bboxes.numpy() == target_gt_bboxes)

    def test_forward_recompute_bbox_from_polygon(self, iseg_entity) -> None:
        # test recompute_bbox = True
        iseg_entity.bboxes = tv_tensors.wrap(torch.tensor([[0.1, 0.1, 0.2, 0.2]]), like=iseg_entity.bboxes)
        iseg_entity.labels = torch.LongTensor([0])
        iseg_entity.masks = tv_tensors.Mask(np.zeros((0, *iseg_entity.img_info.img_shape), dtype=bool))
        target_gt_bboxes = np.array([[0.0, 0.0, 7.0, 7.0]], dtype=np.float32)
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=True,
        )

        results = transform(deepcopy(iseg_entity))

        assert np.all(results.bboxes.numpy() == target_gt_bboxes)

    def test_forward_bbox_clip_border_false(self, det_entity) -> None:
        # test bbox_clip_border = False
        det_entity.bboxes = tv_tensors.wrap(torch.tensor([[0.1, 0.1, 0.2, 0.2]]), like=det_entity.bboxes)
        det_entity.labels = torch.LongTensor([0])
        transform = RandomCrop(
            crop_size=(10, 11),
            allow_negative_crop=False,
            recompute_bbox=True,
            bbox_clip_border=False,
        )

        results = transform(deepcopy(det_entity))

        assert torch.all(results.bboxes == det_entity.bboxes)

    @pytest.mark.parametrize("allow_negative_crop", [True, False])
    def test_forward_allow_negative_crop(self, det_entity, allow_negative_crop: bool) -> None:
        # test the crop does not contain any gt-bbox allow_negative_crop = False
        det_entity.image = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
        det_entity.bboxes = tv_tensors.wrap(torch.zeros((0, 4)), like=det_entity.bboxes)
        det_entity.labels = torch.LongTensor()
        transform = RandomCrop(crop_size=(5, 3), allow_negative_crop=allow_negative_crop)

        results = transform(deepcopy(det_entity))

        if allow_negative_crop:
            assert results.image.shape == transform.crop_size
            assert len(results.bboxes) == len(det_entity.bboxes) == 0
        else:
            assert results is None

    def test_repr(self):
        crop_type = "absolute"
        crop_size = (10, 5)
        allow_negative_crop = False
        recompute_bbox = True
        bbox_clip_border = False
        transform = RandomCrop(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border,
        )
        assert (
            repr(transform) == f"RandomCrop(crop_size={crop_size}, crop_type={crop_type}, "
            f"allow_negative_crop={allow_negative_crop}, "
            f"recompute_bbox={recompute_bbox}, "
            f"bbox_clip_border={bbox_clip_border}, "
            f"is_numpy_to_tvtensor=False)"
        )


class TestFilterAnnotations:
    @pytest.fixture()
    def iseg_entity(self) -> InstanceSegDataEntity:
        masks = np.zeros((3, 224, 224))
        masks[..., 10:20, 10:20] = 1
        masks[..., 20:40, 20:40] = 1
        masks[..., 40:80, 40:80] = 1
        return InstanceSegDataEntity(
            image=np.random.random((224, 224, 3)),
            img_info=ImageInfo(img_idx=0, img_shape=(224, 224), ori_shape=(224, 224)),
            bboxes=tv_tensors.BoundingBoxes(
                np.array([[10, 10, 20, 20], [20, 20, 40, 40], [40, 40, 80, 80]]),
                format="xyxy",
                canvas_size=(224, 224),
            ),
            labels=torch.LongTensor([1, 2, 3]),
            masks=tv_tensors.Mask(masks),
            polygons=[
                Polygon(points=[10, 10, 10, 20, 20, 20, 20, 10]),
                Polygon(points=[20, 20, 20, 40, 40, 40, 40, 20]),
                Polygon(points=[40, 40, 40, 80, 80, 80, 80, 40]),
            ],
        )

    @pytest.mark.parametrize("keep_empty", [True, False])
    def test_forward_keep_empty_by_box(self, iseg_entity, keep_empty: bool) -> None:
        transform = FilterAnnotations(min_gt_bbox_wh=(50, 50), keep_empty=keep_empty, by_box=True)

        results = transform(deepcopy(iseg_entity))

        if keep_empty:
            assert np.all(results.image == iseg_entity.image)
            assert torch.all(results.bboxes == iseg_entity.bboxes)
            assert torch.all(results.masks == iseg_entity.masks)
        else:
            assert results.bboxes.shape[0] == 0
            assert results.masks.shape[0] == 0
            assert len(results.polygons) == 0

    @pytest.mark.parametrize("keep_empty", [True, False])
    def test_forward_keep_empty_by_mask(self, iseg_entity, keep_empty: bool) -> None:
        transform = FilterAnnotations(min_gt_mask_area=2500, keep_empty=keep_empty, by_box=False, by_mask=True)

        results = transform(deepcopy(iseg_entity))

        if keep_empty:
            assert np.all(results.image == iseg_entity.image)
            assert torch.all(results.bboxes == iseg_entity.bboxes)
            assert torch.all(results.masks == iseg_entity.masks)
        else:
            assert results.bboxes.shape[0] == 0
            assert results.masks.shape[0] == 0
            assert len(results.polygons) == 0

    @pytest.mark.parametrize("keep_empty", [True, False])
    def test_forward_keep_empty_by_polygon(self, iseg_entity, keep_empty: bool) -> None:
        transform = FilterAnnotations(min_gt_mask_area=2500, keep_empty=keep_empty, by_box=False, by_polygon=True)

        results = transform(deepcopy(iseg_entity))

        if keep_empty:
            assert np.all(results.image == iseg_entity.image)
            assert torch.all(results.bboxes == iseg_entity.bboxes)
            assert torch.all(results.masks == iseg_entity.masks)
        else:
            assert results.bboxes.shape[0] == 0
            assert results.masks.shape[0] == 0
            assert len(results.polygons) == 0

    def test_forward(self, iseg_entity) -> None:
        # test filter annotations
        transform = FilterAnnotations(min_gt_bbox_wh=(15, 15))

        results = transform(deepcopy(iseg_entity))

        assert torch.all(results.labels == torch.LongTensor([2, 3]))
        assert torch.all(results.bboxes == torch.tensor([[20, 20, 40, 40], [40, 40, 80, 80]]))
        assert len(results.masks) == 2
        assert len(results.polygons) == 2

    def test_repr(self):
        transform = FilterAnnotations(
            min_gt_bbox_wh=(1, 1),
            keep_empty=False,
        )
        assert (
            repr(transform) == "FilterAnnotations(min_gt_bbox_wh=(1, 1), keep_empty=False, is_numpy_to_tvtensor=False)"
        )


class TestTopdownAffine:
    @pytest.fixture()
    def keypoint_det_entity(self) -> KeypointDetDataEntity:
        return KeypointDetDataEntity(
            image=np.random.randint(0, 255, size=(10, 10, 3), dtype=np.uint8),
            img_info=ImageInfo(img_idx=0, img_shape=(10, 10), ori_shape=(10, 10)),
            bboxes=tv_tensors.BoundingBoxes(
                np.array([[0, 0, 7, 7]], dtype=np.float32),
                format="xyxy",
                canvas_size=(10, 10),
            ),
            labels=torch.LongTensor([0]),
            keypoints=tv_tensors.TVTensor(np.array([[0, 4], [4, 2], [2, 6], [6, 0]])),
            keypoints_visible=tv_tensors.TVTensor(np.array([1, 1, 1, 0])),
            bbox_info=BboxInfo(center=np.array([5, 5]), scale=np.array([10, 10]), rotation=0),
        )

    def test_forward(self, keypoint_det_entity) -> None:
        transform = Compose(
            [
                GetBBoxCenterScale(),
                TopdownAffine(input_size=(5, 5)),
            ],
        )
        results = transform(deepcopy(keypoint_det_entity))

        assert np.array_equal(results.bbox_info.center, np.array([3.5, 3.5]))
        assert np.array_equal(results.bbox_info.scale, np.array([8.75, 8.75]))
        assert results.keypoints.shape == (4, 2)
