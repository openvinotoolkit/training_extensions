# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of base data entity."""

import pytest
import torch
import torchvision.transforms.v2 as tvt
import torchvision.transforms.v2.functional as F  # noqa: N812
from otx.core.data.entity.base import ImageType, OTXBatchDataEntity, OTXDataEntity, Points
from otx.core.data.entity.visual_prompting import VisualPromptingDataEntity


class TestOTXDataEntity:
    def test_image_type(
        self,
        fxt_numpy_data_entity,
        fxt_torchvision_data_entity,
    ) -> None:
        assert fxt_numpy_data_entity.image_type == ImageType.NUMPY
        assert fxt_torchvision_data_entity.image_type == ImageType.TV_IMAGE


class TestOTXBatchDataEntity:
    def test_collate_fn(self, mocker, fxt_torchvision_data_entity) -> None:
        mocker.patch.object(OTXDataEntity, "task", return_value="detection")
        mocker.patch.object(OTXBatchDataEntity, "task", return_value="detection")
        data_entities = [
            fxt_torchvision_data_entity,
            fxt_torchvision_data_entity,
            fxt_torchvision_data_entity,
        ]

        data_batch = OTXBatchDataEntity.collate_fn(data_entities)
        assert len(data_batch.imgs_info) == len(data_batch.images)


class TestImageInfo:
    @pytest.fixture(autouse=True)
    def fix_seed(self) -> None:
        torch.manual_seed(3003)

    @pytest.fixture()
    def fxt_resize(self) -> tvt.Resize:
        return tvt.Resize(size=(2, 5))

    @pytest.fixture()
    def fxt_random_resize(self) -> tvt.RandomResize:
        return tvt.RandomResize(min_size=2, max_size=5)

    @pytest.mark.parametrize("fxt_transform", ["fxt_resize", "fxt_random_resize"])
    def test_resize(
        self,
        fxt_torchvision_data_entity: OTXDataEntity,
        fxt_transform: str,
        request: pytest.FixtureRequest,
    ) -> None:
        transform = request.getfixturevalue(fxt_transform)
        transformed = transform(fxt_torchvision_data_entity)

        assert transformed.image.shape[1:] == transformed.img_info.img_shape
        assert fxt_torchvision_data_entity.image.shape[1:] == transformed.img_info.ori_shape

        scale_factor = (
            transformed.image.shape[1] / fxt_torchvision_data_entity.image.shape[1],
            transformed.image.shape[2] / fxt_torchvision_data_entity.image.shape[2],
        )
        assert scale_factor == transformed.img_info.scale_factor

    @pytest.fixture()
    def fxt_random_crop(self) -> tvt.RandomCrop:
        return tvt.RandomCrop(size=(2, 5))

    @pytest.fixture()
    def fxt_random_resized_crop(self) -> tvt.RandomResizedCrop:
        return tvt.RandomResizedCrop(size=(2, 5))

    @pytest.fixture()
    def fxt_center_crop(self) -> tvt.CenterCrop:
        return tvt.CenterCrop(size=(2, 5))

    @pytest.mark.parametrize(
        "fxt_transform",
        ["fxt_random_crop", "fxt_random_resized_crop", "fxt_center_crop"],
    )
    def test_crop(
        self,
        fxt_torchvision_data_entity: OTXDataEntity,
        fxt_transform: str,
        request: pytest.FixtureRequest,
    ) -> None:
        transform = request.getfixturevalue(fxt_transform)
        transformed = transform(fxt_torchvision_data_entity)

        assert transformed.image.shape[1:] == transformed.img_info.img_shape
        assert fxt_torchvision_data_entity.image.shape[1:] == transformed.img_info.ori_shape
        assert transformed.img_info.scale_factor is None

    def test_pad(
        self,
        fxt_torchvision_data_entity: OTXDataEntity,
    ) -> None:
        transform = tvt.Pad(padding=(1, 2, 3, 4))
        transformed = transform(fxt_torchvision_data_entity)

        assert transformed.image.shape[1:] == transformed.img_info.img_shape
        assert fxt_torchvision_data_entity.image.shape[1:] == transformed.img_info.ori_shape
        assert transformed.img_info.padding == (1, 2, 3, 4)

    def test_normalize(
        self,
        fxt_torchvision_data_entity: OTXDataEntity,
    ) -> None:
        mean = (100, 101, 102)
        std = (1, 2, 3)
        transform = tvt.Normalize(mean=mean, std=std)
        transformed = transform(fxt_torchvision_data_entity)

        assert transformed.img_info.normalized
        assert transformed.img_info.norm_mean == mean
        assert transformed.img_info.norm_std == std

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Test only if CUDA is available.")
    def test_to_cuda(
        self,
        fxt_torchvision_data_entity: OTXDataEntity,
    ) -> None:
        cuda_img_info = fxt_torchvision_data_entity.img_info.to(device="cuda")
        # Do not lose its meta info although calling `Tensor.to(device="cuda")`
        assert fxt_torchvision_data_entity.img_info.img_shape == cuda_img_info.img_shape


class TestPoints:
    def test_resize(self, fxt_visual_prompting_data_entity: VisualPromptingDataEntity) -> None:
        transform = tvt.Resize(size=(3, 5))
        results = transform(fxt_visual_prompting_data_entity)

        assert isinstance(results.points, Points)
        assert results.points.canvas_size == tuple(transform.size)
        assert results.points.canvas_size == results.img_info.img_shape

        assert str(results.points) == "Points([3.5000, 2.1000], canvas_size=(3, 5))"

    def test_pad(self, fxt_visual_prompting_data_entity: VisualPromptingDataEntity) -> None:
        transform = tvt.Pad(padding=(1, 2, 3, 4))
        results = transform(fxt_visual_prompting_data_entity)

        assert results.points.canvas_size == results.image[1].shape
        assert torch.all(
            results.points == fxt_visual_prompting_data_entity.points + torch.tensor(transform.padding[:2]),
        )

        assert str(results.points) == "Points([8., 9.], canvas_size=(16, 14))"

    def test_get_size(self, fxt_visual_prompting_data_entity: VisualPromptingDataEntity) -> None:
        results = F.get_size(fxt_visual_prompting_data_entity.points)

        assert results == [10, 10]
